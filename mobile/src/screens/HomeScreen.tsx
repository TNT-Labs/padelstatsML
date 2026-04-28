/**
 * Schermata principale: registra/seleziona video → mostra progresso → mostra stats.
 * Per brevità del MVP, usa expo-image-picker per video selection (no camera live).
 * In versione completa: schermata dedicata con expo-camera + setup wizard.
 */
import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useMatchAnalysis } from '../services/useMatchAnalysis';

export default function HomeScreen() {
  const { state, analyze, reset } = useMatchAnalysis();
  const [title, setTitle] = useState('Match');

  async function pickAndAnalyze() {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permesso negato');
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
      <Text style={styles.title}>Padel Stats</Text>

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
          <Text style={styles.hint}>
            Inquadratura richiesta: posizione fissa, elevata, dietro al campo,
            tutto il campo visibile.
          </Text>
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

function phaseLabel(p: string): string {
  switch (p) {
    case 'creating': return 'Creazione match...';
    case 'uploading': return 'Caricamento video...';
    case 'processing': return 'Analisi in corso...';
    default: return p;
  }
}

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
          <Text>
            Smash: {p.shots.smash} • Volée: {p.shots.volley} • Bandeja: {p.shots.bandeja}
          </Text>
        </View>
      ))}

      <Pressable style={styles.button} onPress={onReset}>
        <Text style={styles.buttonText}>Nuova analisi</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, paddingTop: 60, gap: 16 },
  title: { fontSize: 28, fontWeight: 'bold' },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 16, gap: 12, elevation: 2 },
  label: { fontSize: 14, color: '#374151' },
  input: { borderWidth: 1, borderColor: '#d1d5db', borderRadius: 8, padding: 12, fontSize: 16 },
  button: { backgroundColor: '#16a34a', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  hint: { fontSize: 12, color: '#6b7280', fontStyle: 'italic' },
  statusTitle: { fontSize: 18, fontWeight: '600' },
  progressBar: { height: 8, backgroundColor: '#e5e7eb', borderRadius: 4, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#16a34a' },
  progressText: { textAlign: 'center', color: '#374151' },
  error: { color: '#dc2626', fontWeight: '600' },
  playerCard: { borderTopWidth: 1, borderTopColor: '#e5e7eb', paddingTop: 8, gap: 4 },
  playerTitle: { fontWeight: '600', fontSize: 16 },
});
