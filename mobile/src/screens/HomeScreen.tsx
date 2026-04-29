/**
 * Schermata principale: registra/seleziona video → mostra progresso → mostra stats.
 * Mostra la guida al posizionamento camera al primo avvio (persistita via AsyncStorage).
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';
import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useMatchAnalysis } from '../services/useMatchAnalysis';

const GUIDE_SEEN_KEY = '@padel_camera_guide_seen';

export default function HomeScreen() {
  const { state, analyze, reset } = useMatchAnalysis();
  const [title, setTitle] = useState('Match');
  const [showGuide, setShowGuide] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(GUIDE_SEEN_KEY).then(val => {
      if (!val) setShowGuide(true);
    });
  }, []);

  const dismissGuide = useCallback(async (dontShowAgain: boolean) => {
    setShowGuide(false);
    if (dontShowAgain) await AsyncStorage.setItem(GUIDE_SEEN_KEY, '1');
  }, []);

  async function pickAndAnalyze() {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permesso negato', 'Autorizza l\'accesso alla libreria video nelle impostazioni.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      videoMaxDuration: 7200,
    });
    if (result.canceled) return;
    await analyze(result.assets[0].uri, title);
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.appTitle}>Padel Stats</Text>

      <CameraGuideModal visible={showGuide} onDismiss={dismissGuide} />

      {state.phase === 'idle' && (
        <View style={styles.card}>
          <Text style={styles.label}>Titolo partita</Text>
          <TextInput
            style={styles.input}
            value={title}
            onChangeText={setTitle}
            placeholder="es. Mercoledì sera"
          />
          <Pressable style={styles.button} onPress={pickAndAnalyze}>
            <Text style={styles.buttonText}>Carica video partita</Text>
          </Pressable>
          <Pressable onPress={() => setShowGuide(true)}>
            <Text style={styles.guideLink}>Come posizionare la camera?</Text>
          </Pressable>
        </View>
      )}

      {(state.phase === 'creating' || state.phase === 'uploading' || state.phase === 'processing') && (
        <View style={styles.card}>
          <ActivityIndicator size="large" color="#16a34a" />
          <Text style={styles.statusTitle}>{phaseLabel(state.phase)}</Text>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${state.progress * 100}%` }]} />
          </View>
          <Text style={styles.progressText}>{Math.round(state.progress * 100)}%</Text>
        </View>
      )}

      {state.phase === 'done' && state.stats && (
        <StatsView stats={state.stats} onReset={reset} />
      )}

      {state.phase === 'error' && (
        <View style={styles.card}>
          <Text style={styles.error}>Errore: {state.error}</Text>
          <Pressable style={styles.button} onPress={reset}>
            <Text style={styles.buttonText}>Riprova</Text>
          </Pressable>
        </View>
      )}
    </ScrollView>
  );
}

// ── Camera guide modal ────────────────────────────────────────────────────────

const GUIDE_STEPS = [
  { icon: '📍', text: 'Posizione fissa — usa un treppiede o appoggia il telefono su una superficie stabile. Qualsiasi movimento rovina l\'analisi.' },
  { icon: '📐', text: 'Altezza minima 3–4 m — tribuna, balcone, vetro superiore. Devi vedere l\'intero campo dall\'alto.' },
  { icon: '🎯', text: 'Campo intero visibile — tutte e 4 le linee di fondo e le 2 reti laterali devono essere nell\'inquadratura.' },
  { icon: '🔆', text: 'Buona illuminazione — evita il controluce. Ideale: luce del giorno diffusa o campo al coperto ben illuminato.' },
  { icon: '▶️', text: 'Inizia a registrare prima della battuta — lascia almeno 5 secondi di campo vuoto all\'inizio.' },
];

function CameraGuideModal({
  visible,
  onDismiss,
}: {
  visible: boolean
  onDismiss: (dontShowAgain: boolean) => void
}) {
  return (
    <Modal visible={visible} animationType="slide" transparent presentationStyle="overFullScreen">
      <View style={guide.overlay}>
        <View style={guide.sheet}>
          <Text style={guide.title}>Posizionamento camera</Text>
          <Text style={guide.subtitle}>Segui queste indicazioni per ottenere statistiche accurate.</Text>

          {/* Court diagram */}
          <View style={guide.diagram}>
            <Text style={guide.diagramText}>{'📷\n  ↓\n┌────────┐\n│  net   │\n├────────┤\n│  net   │\n└────────┘'}</Text>
            <Text style={guide.diagramCaption}>Vista dall'alto — camera sopraelevata dietro al campo</Text>
          </View>

          {GUIDE_STEPS.map((step, i) => (
            <View key={i} style={guide.step}>
              <Text style={guide.stepIcon}>{step.icon}</Text>
              <Text style={guide.stepText}>{step.text}</Text>
            </View>
          ))}

          <Pressable style={guide.btnPrimary} onPress={() => onDismiss(true)}>
            <Text style={guide.btnPrimaryText}>Ho capito — non mostrare più</Text>
          </Pressable>
          <Pressable style={guide.btnSecondary} onPress={() => onDismiss(false)}>
            <Text style={guide.btnSecondaryText}>Chiudi</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

// ── Stats summary ─────────────────────────────────────────────────────────────

function StatsView({ stats, onReset }: { stats: any; onReset: () => void }) {
  return (
    <View style={styles.card}>
      <Text style={styles.statusTitle}>Risultati</Text>
      <Text style={styles.label}>Rally totali: {stats.rallies_count}</Text>
      <Text style={styles.label}>Colpi totali: {stats.total_shots}</Text>

      {Object.entries(stats.per_player).map(([pid, p]: [string, any]) => (
        <View key={pid} style={styles.playerCard}>
          <Text style={styles.playerTitle}>Giocatore {pid}</Text>
          <Text>Distanza: {p.distance_m} m</Text>
          <Text>Vincenti: {p.winners} | Errori: {p.errors}</Text>
          <Text>Smash: {p.shots.smash} • Volée: {p.shots.volley} • Bandeja: {p.shots.bandeja}</Text>
        </View>
      ))}

      <Pressable style={styles.button} onPress={onReset}>
        <Text style={styles.buttonText}>Nuova analisi</Text>
      </Pressable>
    </View>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function phaseLabel(p: string): string {
  switch (p) {
    case 'creating':   return 'Creazione match…';
    case 'uploading':  return 'Caricamento video…';
    case 'processing': return 'Analisi in corso…';
    default:           return p;
  }
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container:    { padding: 20, paddingTop: 60, gap: 16 },
  appTitle:     { fontSize: 28, fontWeight: 'bold' },
  card:         { backgroundColor: '#fff', borderRadius: 12, padding: 16, gap: 12, elevation: 2 },
  label:        { fontSize: 14, color: '#374151' },
  input:        { borderWidth: 1, borderColor: '#d1d5db', borderRadius: 8, padding: 12, fontSize: 16 },
  button:       { backgroundColor: '#16a34a', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText:   { color: '#fff', fontWeight: '600', fontSize: 16 },
  guideLink:    { fontSize: 13, color: '#2563eb', textAlign: 'center', textDecorationLine: 'underline' },
  statusTitle:  { fontSize: 18, fontWeight: '600' },
  progressBar:  { height: 8, backgroundColor: '#e5e7eb', borderRadius: 4, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#16a34a' },
  progressText: { textAlign: 'center', color: '#374151' },
  error:        { color: '#dc2626', fontWeight: '600' },
  playerCard:   { borderTopWidth: 1, borderTopColor: '#e5e7eb', paddingTop: 8, gap: 4 },
  playerTitle:  { fontWeight: '600', fontSize: 16 },
});

const guide = StyleSheet.create({
  overlay:         { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  sheet:           { backgroundColor: '#fff', borderTopLeftRadius: 20, borderTopRightRadius: 20, padding: 24, gap: 12, maxHeight: '90%' },
  title:           { fontSize: 22, fontWeight: 'bold', textAlign: 'center' },
  subtitle:        { fontSize: 14, color: '#6b7280', textAlign: 'center' },
  diagram:         { backgroundColor: '#f0fdf4', borderRadius: 8, padding: 12, alignItems: 'center' },
  diagramText:     { fontFamily: 'monospace', fontSize: 13, color: '#15803d', lineHeight: 18 },
  diagramCaption:  { fontSize: 11, color: '#6b7280', marginTop: 4 },
  step:            { flexDirection: 'row', gap: 10, alignItems: 'flex-start' },
  stepIcon:        { fontSize: 20, width: 28 },
  stepText:        { flex: 1, fontSize: 14, color: '#374151', lineHeight: 20 },
  btnPrimary:      { backgroundColor: '#16a34a', padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 8 },
  btnPrimaryText:  { color: '#fff', fontWeight: '600', fontSize: 15 },
  btnSecondary:    { padding: 10, alignItems: 'center' },
  btnSecondaryText:{ color: '#6b7280', fontSize: 14 },
});
