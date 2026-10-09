"""Accounts: sign-in, sessions, CSRF, ownership, administration, chunked
upload and the base path. The rule under test everywhere: only a user created
by an administrator reaches any data, and only their own."""
from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from conftest import PASSWORD, ensure_user, signed_in



def _name(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _anonymous(app=None, **headers) -> AsyncClient:
    from app.main import app as default_app

    return AsyncClient(
        transport=ASGITransport(app=app or default_app), base_url="http://testserver",
        headers={"X-Padel": "1", **headers},
    )


async def _match_of(client) -> str:
    response = await client.post("/api/matches", json={"title": "Mia"})
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


# ── Sign-in and sessions ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_without_a_session_nothing_is_reachable_but_health():
    async with _anonymous() as c:
        for path in ("/api/matches", "/api/presets", "/api/admin/users", "/api/auth/me"):
            assert (await c.get(path)).status_code == 401, path
        assert (await c.post("/api/matches", json={"title": "x"})).status_code == 401
        assert (await c.get("/api/health")).status_code in (200, 503)


@pytest.mark.asyncio
async def test_login_sets_a_strict_http_only_cookie_and_logout_ends_it():
    name = _name("u")
    ensure_user(name)
    async with _anonymous() as c:
        response = await c.post("/api/auth/login", json={"username": name.upper(), "password": PASSWORD})
        assert response.status_code == 200
        assert response.json()["username"] == name
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
        assert (await c.get("/api/matches")).status_code == 200

        assert (await c.post("/api/auth/logout")).status_code == 204
        c.cookies.clear()
        assert (await c.get("/api/matches")).status_code == 401


@pytest.mark.asyncio
async def test_a_logged_out_token_is_dead_even_if_replayed():
    name = _name("u")
    ensure_user(name)
    async with _anonymous() as c:
        await c.post("/api/auth/login", json={"username": name, "password": PASSWORD})
        token = c.cookies.get("padel_session")
        await c.post("/api/auth/logout")
    async with _anonymous() as thief:
        thief.cookies.set("padel_session", token)
        assert (await thief.get("/api/matches")).status_code == 401


@pytest.mark.asyncio
async def test_wrong_credentials_get_one_message_for_every_cause():
    name = _name("u")
    ensure_user(name)
    ensure_user(off := _name("off"), active=False)
    async with _anonymous() as c:
        wrong = await c.post("/api/auth/login", json={"username": name, "password": "Sbagliata123"})
        missing = await c.post("/api/auth/login", json={"username": "nessuno", "password": PASSWORD})
        disabled = await c.post("/api/auth/login", json={"username": off, "password": PASSWORD})
    assert wrong.status_code == missing.status_code == disabled.status_code == 401
    assert wrong.json() == missing.json() == disabled.json()


@pytest.mark.asyncio
async def test_the_account_locks_after_repeated_failures_and_an_admin_unlocks_it():
    from app.core.security import login_limiter

    login_limiter.reset()
    name = _name("u")
    user_id = ensure_user(name)
    async with _anonymous() as c:
        for _ in range(5):
            await c.post("/api/auth/login", json={"username": name, "password": "Sbagliata123"})
        locked = await c.post("/api/auth/login", json={"username": name, "password": PASSWORD})
        assert locked.status_code == 401, "il blocco deve valere anche con la password giusta"

    admin = await signed_in(_name("adm"), "admin")
    async with admin:
        users = {u["id"]: u for u in (await admin.get("/api/admin/users")).json()}
        assert users[user_id]["locked"] is True
        assert (await admin.post(f"/api/admin/users/{user_id}/unlock")).json()["locked"] is False
    async with _anonymous() as c:
        assert (await c.post("/api/auth/login", json={"username": name, "password": PASSWORD})).status_code == 200


@pytest.mark.asyncio
async def test_too_many_failures_from_one_address_are_throttled():
    from app.core.security import login_limiter

    login_limiter.reset()
    async with _anonymous() as c:
        codes = [
            (await c.post("/api/auth/login", json={"username": "x" + str(i), "password": "Nope12345a"})).status_code
            for i in range(21)
        ]
    login_limiter.reset()
    assert codes[:20] == [401] * 20 and codes[20] == 429


@pytest.mark.asyncio
async def test_an_idle_session_expires():
    from sqlalchemy import update

    from app.core.database import sync_session
    from app.core.security import utcnow
    from app.models import AuthSession

    name = _name("u")
    user_id = ensure_user(name)
    async with await signed_in(name) as c:
        with sync_session() as session:
            session.execute(
                update(AuthSession).where(AuthSession.user_id == user_id)
                .values(last_seen_at=utcnow() - timedelta(hours=13))
            )
        assert (await c.get("/api/matches")).status_code == 401


# ── Password change ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_temporary_password_must_be_replaced_before_anything_else():
    name = _name("u")
    async with await signed_in(name, must_change=True) as c:
        assert (await c.get("/api/matches")).status_code == 403
        assert (await c.get("/api/auth/me")).json()["must_change_password"] is True

        for weak, reason in (("corta1", "almeno"), ("solamentelettere", "cifre"), (name + "123", "nome utente")):
            response = await c.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": weak})
            assert response.status_code == 400 and reason in response.json()["detail"], weak
        wrong = await c.post("/api/auth/password", json={"current_password": "Altra12345", "new_password": "Nuova12345x"})
        assert wrong.status_code == 400

        ok = await c.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": "Nuova12345x"})
        assert ok.status_code == 200 and ok.json()["must_change_password"] is False
        assert (await c.get("/api/matches")).status_code == 200


@pytest.mark.asyncio
async def test_changing_password_signs_out_the_other_sessions():
    name = _name("u")
    first = await signed_in(name)
    second = _anonymous()
    async with first, second:
        await second.post("/api/auth/login", json={"username": name, "password": PASSWORD})
        await first.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": "Nuova12345x"})
        assert (await first.get("/api/matches")).status_code == 200
        assert (await second.get("/api/matches")).status_code == 401


# ── CSRF and headers ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_writes_need_the_custom_header_and_a_matching_origin():
    async with await signed_in(_name("u")) as c:
        bare = await c.post("/api/matches", json={"title": "x"}, headers={"X-Padel": "0"})
        assert bare.status_code == 403
        foreign = await c.post("/api/matches", json={"title": "x"}, headers={"Origin": "https://evil.example"})
        assert foreign.status_code == 403
        same = await c.post("/api/matches", json={"title": "x"}, headers={"Origin": "http://testserver"})
        assert same.status_code == 201
        # Reads are not affected: <video> and <img> cannot send headers.
        assert (await c.get("/api/matches", headers={"X-Padel": "0"})).status_code == 200


@pytest.mark.asyncio
async def test_responses_carry_the_security_headers():
    async with await signed_in(_name("u")) as c:
        response = await c.get("/api/matches")
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "no-cache" in response.headers["cache-control"]


# ── Ownership ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_users_see_only_their_matches_and_administrators_all():
    alice, bob = _name("alice"), _name("bob")
    a = await signed_in(alice)
    b = await signed_in(bob)
    admin = await signed_in(_name("adm"), "admin")
    async with a, b, admin:
        match_id = await _match_of(a)
        assert match_id in [m["id"] for m in (await a.get("/api/matches")).json()]
        assert match_id not in [m["id"] for m in (await b.get("/api/matches")).json()]
        for method, path in (
            ("GET", f"/api/matches/{match_id}"),
            ("GET", f"/api/matches/{match_id}/video"),
            ("GET", f"/api/matches/{match_id}/upload"),
            ("PATCH", f"/api/matches/{match_id}"),
            ("POST", f"/api/matches/{match_id}/start"),
            ("DELETE", f"/api/matches/{match_id}"),
        ):
            response = await b.request(method, path, json={"title": "presa"} if method == "PATCH" else None)
            assert response.status_code == 404, (method, path, response.status_code)

        listed = {m["id"]: m for m in (await admin.get("/api/matches")).json()}
        assert listed[match_id]["owner"] == alice
        assert (await admin.get(f"/api/matches/{match_id}")).json()["owner"] == alice
        assert (await a.get(f"/api/matches/{match_id}")).json()["owner"] is None


@pytest.mark.asyncio
async def test_presets_are_personal():
    from test_api import GOOD_CORNERS, W, H

    a = await signed_in(_name("alice"))
    b = await signed_in(_name("bob"))
    body = {"name": "Campo 1", "corners_px": GOOD_CORNERS, "frame_width": W, "frame_height": H}
    async with a, b:
        first = await a.post("/api/presets", json=body)
        assert first.status_code == 201
        # The same name is free for someone else.
        assert (await b.post("/api/presets", json=body)).status_code == 201
        assert [p["id"] for p in (await a.get("/api/presets")).json()] == [first.json()["id"]]
        assert (await b.delete(f"/api/presets/{first.json()['id']}")).status_code == 404

        match_id = await _match_of(b)
        from app.core.database import sync_session
        from app.models import Match

        with sync_session() as session:
            match = session.get(Match, match_id)
            match.width, match.height = W, H
        stolen = await b.post(f"/api/matches/{match_id}/calibration", json={"preset_id": first.json()["id"]})
        assert stolen.status_code == 404


# ── Administration ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_administrators_manage_users():
    async with await signed_in(_name("u")) as c:
        assert (await c.get("/api/admin/users")).status_code == 403
        assert (await c.post("/api/admin/users", json={"username": "nuovo"})).status_code == 403
        assert (await c.get("/api/admin/audit")).status_code == 403


@pytest.mark.asyncio
async def test_a_created_user_signs_in_with_the_temporary_password_once():
    name = _name("nuovo")
    async with await signed_in(_name("adm"), "admin") as admin:
        created = await admin.post("/api/admin/users", json={"username": f"  {name.upper()} "})
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["user"]["username"] == name and body["user"]["must_change_password"]
        assert (await admin.post("/api/admin/users", json={"username": name})).status_code == 409
        assert (await admin.post("/api/admin/users", json={"username": "a b"})).status_code == 400

    async with _anonymous() as c:
        login = await c.post("/api/auth/login", json={"username": name, "password": body["temporary_password"]})
        assert login.status_code == 200 and login.json()["must_change_password"] is True
        assert (await c.get("/api/matches")).status_code == 403


@pytest.mark.asyncio
async def test_reset_and_deactivation_close_the_sessions_at_once():
    name = _name("u")
    user = await signed_in(name)
    async with await signed_in(_name("adm"), "admin") as admin, user:
        user_id = (await user.get("/api/auth/me")).json()["id"]
        reset = await admin.post(f"/api/admin/users/{user_id}/reset-password")
        assert reset.status_code == 200
        assert (await user.get("/api/matches")).status_code == 401

        temp = reset.json()["temporary_password"]
        assert (await user.post("/api/auth/login", json={"username": name, "password": temp})).status_code == 200
        assert (await admin.patch(f"/api/admin/users/{user_id}", json={"active": False})).status_code == 200
        assert (await user.get("/api/auth/me")).status_code == 401
        assert (await user.post("/api/auth/login", json={"username": name, "password": temp})).status_code == 401


@pytest.mark.asyncio
async def test_an_administrator_cannot_lock_themselves_out():
    from sqlalchemy import update

    from app.core.database import sync_session
    from app.models import User, UserRole

    name = _name("adm")
    async with await signed_in(name, "admin") as admin:
        me = (await admin.get("/api/auth/me")).json()["id"]
        assert (await admin.patch(f"/api/admin/users/{me}", json={"role": "utente"})).status_code == 409
        assert (await admin.patch(f"/api/admin/users/{me}", json={"active": False})).status_code == 409
        assert (await admin.delete(f"/api/admin/users/{me}")).status_code == 409

        # The last active administrator cannot be removed by anyone.
        other = _name("adm")
        other_id = ensure_user(other, "admin")
        with sync_session() as session:
            session.execute(
                update(User).where(User.role == UserRole.ADMIN, User.id.not_in([me, other_id]))
                .values(active=False)
            )
        assert (await admin.patch(f"/api/admin/users/{other_id}", json={"role": "utente"})).status_code == 200
        async with await signed_in(other, "utente") as demoted:
            assert (await demoted.get("/api/admin/users")).status_code == 403


@pytest.mark.asyncio
async def test_deleting_a_user_deletes_their_matches_and_files(data_dir):
    from test_api import _make_video

    name = _name("u")
    user = await signed_in(name)
    async with await signed_in(_name("adm"), "admin") as admin, user:
        match_id = await _match_of(user)
        video = _make_video(data_dir / f"{match_id}-src.mp4")
        assert (await user.put(f"/api/matches/{match_id}/video", content=video.read_bytes())).status_code == 200
        assert (data_dir / "videos" / f"{match_id}.mp4").exists()

        user_id = (await user.get("/api/auth/me")).json()["id"]
        assert (await admin.delete(f"/api/admin/users/{user_id}")).status_code == 204
        assert (await admin.get(f"/api/matches/{match_id}")).status_code == 404
        assert not (data_dir / "videos" / f"{match_id}.mp4").exists()
        assert not (data_dir / "keyframes" / f"{match_id}.jpg").exists()
        assert (await user.get("/api/matches")).status_code == 401

        events = [(e["event"], e["detail"]) for e in (await admin.get("/api/admin/audit")).json()]
        assert ("utente_eliminato", f"{name} con 1 partita") in events


# ── Chunked upload ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_video_arrives_in_pieces_and_resumes_after_a_lost_answer(tmp_path):
    from test_api import H, W, _make_video

    data = _make_video(tmp_path / "clip.mp4").read_bytes()
    third = len(data) // 3
    async with await signed_in(_name("u")) as c:
        init = (await c.post("/api/matches", json={"title": "A pezzi"})).json()
        match_id = init["match_id"]
        assert init["chunk_size_bytes"] == 32 * 1024 * 1024
        url = f"/api/matches/{match_id}/upload"

        assert (await c.put(url, params={"offset": 0}, content=data[:third])).json()["received_bytes"] == third
        # The same piece again (its answer was lost): refused with the position.
        again = await c.put(url, params={"offset": 0 + 1}, content=data[:third])
        assert again.status_code == 409 and again.headers["x-received-bytes"] == str(third)
        assert (await c.get(url)).json()["received_bytes"] == third

        await c.put(url, params={"offset": third}, content=data[third:2 * third])
        early = await c.post(f"{url}/complete", json={"total_bytes": len(data)})
        assert early.status_code == 409
        await c.put(url, params={"offset": 2 * third}, content=data[2 * third:])

        done = await c.post(f"{url}/complete", json={"total_bytes": len(data)})
        assert done.status_code == 200, done.text
        assert done.json()["status"] == "needs_calibration"
        assert (done.json()["width"], done.json()["height"]) == (W, H)
        # Finished: no more pieces accepted.
        assert (await c.put(url, params={"offset": 0}, content=b"x")).status_code == 409


@pytest.mark.asyncio
async def test_a_piece_over_the_limit_is_refused_and_cut_back():
    from app.core.config import get_settings

    settings = get_settings()
    original = settings.upload_chunk_mb
    settings.upload_chunk_mb = 1
    try:
        async with await signed_in(_name("u")) as c:
            match_id = await _match_of(c)
            url = f"/api/matches/{match_id}/upload"
            assert (await c.put(url, params={"offset": 0}, content=b"a" * 1000)).status_code == 200
            big = await c.put(url, params={"offset": 1000}, content=b"b" * (1024 * 1024 + 1))
            assert big.status_code == 413
            assert (await c.get(url)).json()["received_bytes"] == 1000
    finally:
        settings.upload_chunk_mb = original


# ── Base path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_application_lives_under_its_base_path():
    from app.main import create_app

    app = create_app(base_path="/padel")
    async with _anonymous(app) as c:
        redirect = await c.get("/padel")
        assert redirect.status_code == 308 and redirect.headers["location"] == "/padel/"
        assert (await c.get("/padel/api/health")).status_code in (200, 503)
        assert (await c.get("/padel/api/matches")).status_code == 401
        assert (await c.get("/api/matches")).status_code == 404

        name = _name("u")
        ensure_user(name)
        login = await c.post("/padel/api/auth/login", json={"username": name, "password": PASSWORD})
        assert login.status_code == 200
        assert "path=/padel" in login.headers["set-cookie"].lower()


# ── Schema upgrade ───────────────────────────────────────────────────────────

def test_a_database_from_before_accounts_is_upgraded_in_place(tmp_path):
    from sqlalchemy import create_engine, text

    from app.core.accounts import adopt_orphans, ensure_admin
    from app.core.database import init_schema
    from sqlalchemy.orm import Session

    from sqlalchemy import event

    from app.core.database import _apply_pragmas

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    # The same pragmas as production: foreign keys on is what makes adding a
    # REFERENCES column fragile.
    event.listen(engine, "connect", _apply_pragmas)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE matches (id VARCHAR(36) PRIMARY KEY, title VARCHAR(200), status VARCHAR(17), "
            "progress INTEGER, progress_message VARCHAR(200), error_message TEXT, "
            "video_filename VARCHAR(200), video_size_bytes INTEGER, duration_seconds FLOAT, fps FLOAT, "
            "width INTEGER, height INTEGER, calibration JSON, player_names JSON, "
            "created_at DATETIME, updated_at DATETIME)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE camera_presets (id VARCHAR(36) PRIMARY KEY, name VARCHAR(100) UNIQUE, "
            "corners_px JSON, frame_width INTEGER, frame_height INTEGER, net_px JSON, times_used INTEGER, "
            "created_at DATETIME, updated_at DATETIME)"
        )
        conn.exec_driver_sql(
            "INSERT INTO matches (id, title, status, progress, video_filename, created_at, updated_at) "
            "VALUES ('m1', 'Vecchia', 'completed', 100, 'm1.mp4', '2026-01-01', '2026-01-01')"
        )
        conn.exec_driver_sql(
            "INSERT INTO camera_presets VALUES ('p1', 'Campo 1', '[[0,0],[1,0],[1,1],[0,1]]', 640, 360, "
            "NULL, 3, '2026-01-01', '2026-01-01')"
        )

    init_schema(engine)
    init_schema(engine)          # a second start finds nothing to do
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA user_version").scalar() == 1
        assert conn.exec_driver_sql("SELECT count(*) FROM matches").scalar() == 1
        assert conn.exec_driver_sql("SELECT name, times_used FROM camera_presets").one() == ("Campo 1", 3)

    with Session(engine) as session:
        admin = ensure_admin(session, "Admin", "Iniziale12345")
        assert admin is not None and admin.must_change_password
        admin_id = admin.id
        assert adopt_orphans(session) == 1
        session.commit()
        assert ensure_admin(session, "altro", "Iniziale12345") is None
    with engine.connect() as conn:
        match_owner = conn.exec_driver_sql("SELECT owner_id FROM matches").scalar()
        preset_owner = conn.exec_driver_sql("SELECT owner_id FROM camera_presets").scalar()
        assert match_owner == preset_owner == admin_id
        # Two users can now own a preset with the same name.
        conn.exec_driver_sql(
            "INSERT INTO users (id, username, password_hash, role, active, must_change_password, "
            "failed_attempts, created_at, updated_at) VALUES ('u2', 'b', 'x', 'utente', 1, 0, 0, '2026-01-01', '2026-01-01')"
        )
        conn.exec_driver_sql(
            "INSERT INTO camera_presets (id, owner_id, name, corners_px, frame_width, frame_height, times_used, "
            "created_at, updated_at) VALUES ('p2', 'u2', 'Campo 1', '[]', 640, 360, 0, '2026-01-01', '2026-01-01')"
        )
        conn.commit()


def test_an_invalid_initial_password_does_not_stop_the_start(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.accounts import ensure_admin
    from app.core.database import init_schema

    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    init_schema(engine)
    with Session(engine) as session:
        assert ensure_admin(session, "admin", "corta") is None
        assert ensure_admin(session, "admin", "") is None


def test_the_command_line_recovers_an_administrator():
    from sqlalchemy import select

    from app.core.accounts import reset_admin
    from app.core.database import sync_session
    from app.core.security import verify_password
    from app.models import User

    name = _name("rec")
    ensure_user(name, "utente", active=False)
    with sync_session() as session:
        password = reset_admin(session, name)
    with sync_session() as session:
        user = session.scalar(select(User).where(User.username == name))
        assert user.is_admin and user.active and user.must_change_password
        assert verify_password(password, user.password_hash)


# ── Hardening ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_oversized_body_is_refused_before_anyone_signs_in():
    async with _anonymous() as c:
        big = b'{"username":"' + b"a" * (2 * 1024 * 1024) + b'","password":"x"}'
        response = await c.post("/api/auth/login", content=big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_health_shows_details_only_on_the_machine_itself():
    from app.main import app

    async with _anonymous() as local:
        assert "checks" in (await local.get("/api/health")).json()
    remote = AsyncClient(
        transport=ASGITransport(app=app, client=("203.0.113.7", 4242)), base_url="http://testserver"
    )
    async with remote:
        body = (await remote.get("/api/health")).json()
    assert set(body) == {"status"}


@pytest.mark.asyncio
async def test_guessing_the_current_password_locks_the_account_and_ends_its_sessions():
    from app.core.security import login_limiter

    login_limiter.reset()
    name = _name("u")
    async with await signed_in(name) as stolen:
        for i in range(5):
            response = await stolen.post(
                "/api/auth/password", json={"current_password": f"Tentativo{i}x", "new_password": "Nuova12345xx"}
            )
            assert response.status_code in (400, 401)
        assert (await stolen.get("/api/matches")).status_code == 401
    async with _anonymous() as c:
        locked = await c.post("/api/auth/login", json={"username": name, "password": PASSWORD})
        assert locked.status_code == 401
    login_limiter.reset()


@pytest.mark.asyncio
async def test_uploads_stop_before_the_disk_is_full(monkeypatch):
    from app.api import matches

    async with await signed_in(_name("u")) as c:
        match_id = await _match_of(c)
        monkeypatch.setattr(matches, "free_space_bytes", lambda: 900 * 1024 * 1024)
        piece = await c.put(f"/api/matches/{match_id}/upload", params={"offset": 0}, content=b"x" * 100)
        assert piece.status_code == 507
        whole = await c.put(f"/api/matches/{match_id}/video", content=b"x" * 100)
        assert whole.status_code == 507


@pytest.mark.asyncio
async def test_player_names_have_a_length_limit():
    async with await signed_in(_name("u")) as c:
        match_id = await _match_of(c)
        response = await c.patch(f"/api/matches/{match_id}", json={"player_names": ["x" * 61]})
        assert response.status_code == 422
