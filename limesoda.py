#!/usr/bin/env python3
"""
 ██╗     ██╗███╗   ███╗███████╗███████╗ ██████╗ ██████╗  █████╗
 ██║     ██║████╗ ████║██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔══██╗
 ██║     ██║██╔████╔██║█████╗  ███████╗██║   ██║██║  ██║███████║
 ██║     ██║██║╚██╔╝██║██╔══╝  ╚════██║██║   ██║██║  ██║██╔══██║
 ███████╗██║██║ ╚═╝ ██║███████╗███████║╚██████╔╝██████╔╝██║  ██║
 ╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝

  DAW — Inspired by FL Studio & LMMS
  Requirements: pip install PyQt6 numpy sounddevice soundfile

FEATURES
────────
  Arrangement   Multi-track timeline, clip drag/resize/copy/cut/paste/split/fade
  Beat Editor   Per-track 16-step sequencer with velocity per step
  Piano Roll    MIDI note draw/erase/move/resize, quantise, velocity editing
  Mixer         Per-track fader, pan knob, mute/solo, send-to-master
  FX Chain      Reverb (wet), 3-band EQ, compressor, per-track
  Automation    Volume automation lane with breakpoint editing
  Transport     Play/Pause/Stop/Loop, BPM, time-signature display
  Export        WAV (built-in), OGG / FLAC (soundfile)
  Undo/Redo     Ctrl+Z / Ctrl+Y
  Copy/Paste    Ctrl+C / Ctrl+X / Ctrl+V  or right-click menu
  Tools         Select (V), Cut (C)
  Zoom          Slider, ±buttons, Ctrl+Scroll
"""

import sys, os, math, time, struct, threading, copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QScrollArea, QLabel, QPushButton, QSlider, QSpinBox,
    QFileDialog, QFrame, QMessageBox, QMenu, QInputDialog,
    QScrollBar, QDialog, QProgressBar, QComboBox, QTabWidget,
    QCheckBox, QGroupBox, QGridLayout, QAbstractScrollArea,
)
from PyQt6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QRect, QPoint,
    pyqtSignal, QThread, QMimeData, QSize, QObject,
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QPixmap, QDrag, QAction,
)

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False

# ══════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════
BG      = QColor("#0A0A0C")
PANEL   = QColor("#111116")
PANEL2  = QColor("#16161E")
PANEL3  = QColor("#1C1C28")
BORDER  = QColor("#2A2A3A")
TEXT    = QColor("#C8CCD8")
TEXTDIM = QColor("#555568")
ACCENT  = QColor("#00F5A0")
ACCENT2 = QColor("#00D4FF")
ACCENT3 = QColor("#FF6B35")
ACCENT4 = QColor("#B44FFF")

TRACK_COLORS = [
    "#00F5A0","#00D4FF","#FF6B35","#B44FFF",
    "#FFD700","#FF4F8B","#4FFFB0","#7EB8FF",
    "#FFAA44","#FF5555","#55FFDD","#CC88FF",
]

TRACK_H      = 70
HDR_W        = 230
RULER_H      = 28
DEFAULT_BPX  = 60
BEAT_STEPS   = 16
SR           = 44100
NOTE_H       = 10
PIANO_W      = 56
TOTAL_NOTES  = 84          # 7 octaves C0–B6
NOTE_NAMES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# ══════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════
_counters = {"s":0,"t":0,"c":0,"n":0}
def _uid(k):
    _counters[k] += 1
    return _counters[k]


class Sample:
    def __init__(self, path: str):
        self.id       = _uid("s")
        self.path     = path
        self.name     = os.path.splitext(os.path.basename(path))[0]
        self.ext      = os.path.splitext(path)[1].lstrip(".").upper()
        self.data: Optional[np.ndarray] = None
        self.sr: int  = SR
        self.duration: float = 0.0
        self._load()

    def _load(self):
        if not AUDIO_OK:
            return
        try:
            d, sr = sf.read(self.path, dtype="float32", always_2d=True)
            self.data, self.sr, self.duration = d, sr, len(d) / sr
        except Exception as e:
            print(f"[WARN] {self.path}: {e}")

    @classmethod
    def from_array(cls, name: str, data: np.ndarray, sr: int = SR):
        s = object.__new__(cls)
        s.id = _uid("s"); s.path = ""; s.name = name; s.ext = "WAV"
        s.data = data.astype(np.float32); s.sr = sr; s.duration = len(data) / sr
        return s


@dataclass
class AutoPoint:
    beat:  float
    value: float   # 0..1


@dataclass
class MidiNote:
    pitch:    int    # 0=C0 .. 83=B6
    start:    float  # beats
    length:   float  # beats
    velocity: int    # 0..127
    id: int = field(default_factory=lambda: _uid("n"))


class Track:
    _ci = 0
    def __init__(self, name: str):
        self.id    = _uid("t")
        self.name  = name
        self.color = TRACK_COLORS[Track._ci % len(TRACK_COLORS)]
        Track._ci += 1
        self.volume: float = 1.0
        self.pan:    float = 0.0      # -1..1
        self.muted:  bool  = False
        self.solo:   bool  = False
        self.clips:  List[int] = []
        # Beat sequencer
        self.steps:     List[bool] = [False] * BEAT_STEPS
        self.step_vel:  List[int]  = [100]   * BEAT_STEPS
        self.step_sample: Optional[Sample] = None
        # Piano roll MIDI
        self.midi_notes: List[MidiNote] = []
        # Automation
        self.vol_auto: List[AutoPoint] = []
        # FX
        self.fx_reverb:  float = 0.0   # 0..1 wet
        self.fx_eq_low:  float = 0.0   # dB
        self.fx_eq_mid:  float = 0.0
        self.fx_eq_hi:   float = 0.0
        self.fx_comp:    bool  = False


class Clip:
    def __init__(self, tid: int, sample: Sample, start: float, length: float):
        self.id           = _uid("c")
        self.track_id     = tid
        self.sample       = sample
        self.start_beat   = start
        self.length_beats = length
        self.gain:     float = 1.0
        self.fade_in:  float = 0.0  # beats
        self.fade_out: float = 0.0


class Project:
    def __init__(self):
        self.name     = "Untitled"
        self.samples: Dict[int, Sample] = {}
        self.tracks:  Dict[int, Track]  = {}
        self.clips:   Dict[int, Clip]   = {}
        self.bpm:     int  = 120
        self.ts_num:  int  = 4
        self.ts_den:  int  = 4
        self._undo_stack: List[tuple] = []
        self._redo_stack: List[tuple] = []

    def add_sample(self, path: str) -> Sample:
        s = Sample(path); self.samples[s.id] = s; return s

    def add_track(self, name: str = "") -> Track:
        t = Track(name or f"Track {len(self.tracks)+1}")
        self.tracks[t.id] = t; return t

    def add_clip(self, tid: int, sample: Sample,
                 start: float, length: float = 0.0) -> Clip:
        if length == 0.0:
            length = max(0.5, sample.duration * self.bps())
        c = Clip(tid, sample, start, length)
        self.clips[c.id] = c
        self.tracks[tid].clips.append(c.id)
        return c

    def remove_clip(self, cid: int):
        c = self.clips.pop(cid, None)
        if c:
            t = self.tracks.get(c.track_id)
            if t: t.clips = [x for x in t.clips if x != cid]

    def remove_track(self, tid: int):
        t = self.tracks.pop(tid, None)
        if t:
            for cid in list(t.clips): self.clips.pop(cid, None)

    def split_clip(self, cid: int, cut_beat: float):
        c = self.clips.get(cid)
        if not c: return None
        off = cut_beat - c.start_beat
        if off <= 0.05 or off >= c.length_beats - 0.05: return None
        orig = c.length_beats; c.length_beats = off
        rs = c.sample
        if c.sample.data is not None:
            f0 = int(off * self.spb() * c.sample.sr)
            rd = c.sample.data[f0:]
            if len(rd) > 0:
                rs = Sample.from_array(c.sample.name, rd, c.sample.sr)
                self.samples[rs.id] = rs
        rc = Clip(c.track_id, rs, cut_beat, orig - off)
        rc.gain = c.gain
        self.clips[rc.id] = rc
        t = self.tracks.get(c.track_id)
        if t: t.clips.append(rc.id)
        return c, rc

    def max_beat(self) -> float:
        if not self.clips: return 32.0
        return max(c.start_beat + c.length_beats for c in self.clips.values()) + 4

    def bps(self): return self.bpm / 60.0
    def spb(self): return 60.0 / self.bpm

    # ── undo/redo ──
    def push_undo(self, undo_fn, redo_fn, name=""):
        self._undo_stack.append((name, undo_fn, redo_fn))
        self._redo_stack.clear()
        if len(self._undo_stack) > 60: self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack: return
        name, u, r = self._undo_stack.pop()
        u(); self._redo_stack.append((name, u, r))

    def redo(self):
        if not self._redo_stack: return
        name, u, r = self._redo_stack.pop()
        r(); self._undo_stack.append((name, u, r))


# ══════════════════════════════════════════════════════════
#  WAVEFORM CACHE
# ══════════════════════════════════════════════════════════
_peaks_cache: Dict = {}

def get_peaks(sample: Sample, width: int) -> np.ndarray:
    key = (sample.id, width)
    if key in _peaks_cache: return _peaks_cache[key]
    if sample.data is None or width <= 0:
        return np.zeros((max(1, width), 2), dtype=np.float32)
    mono = sample.data[:, 0] if sample.data.ndim > 1 else sample.data
    total = len(mono); step = max(1, total // width)
    peaks = np.zeros((width, 2), dtype=np.float32)
    for i in range(width):
        s = i * step; e = min(s + step, total)
        if s >= total: break
        chunk = mono[s:e]; peaks[i] = [chunk.min(), chunk.max()]
    _peaks_cache[key] = peaks
    return peaks


# ══════════════════════════════════════════════════════════
#  AUDIO ENGINE
# ══════════════════════════════════════════════════════════
class AudioEngine(QObject):
    tick = pyqtSignal(float)

    def __init__(self, proj: Project):
        super().__init__()
        self.proj     = proj
        self.playing  = False
        self.looping  = False
        self.position = 0.0
        self._stream  = None
        self._mix: Optional[np.ndarray] = None
        self._buf_pos = 0
        self._lock    = threading.Lock()
        self._wall    = 0.0
        self._beat0   = 0.0
        self._timer   = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._poll)

    # ── position polling ──
    def _poll(self):
        if not self.playing: return
        beat = self._beat0 + (time.perf_counter() - self._wall) * self.proj.bps()
        mb   = self.proj.max_beat()
        if beat >= mb:
            if self.looping: self.seek(0.0); self._start_stream(); return
            else: self.stop(); self.position = 0.0; self.tick.emit(0.0); return
        self.position = beat; self.tick.emit(beat)

    # ── FX processors ──
    @staticmethod
    def _eq(data: np.ndarray, sr: int, lo: float, mid: float, hi: float) -> np.ndarray:
        if lo == mid == hi == 0.0: return data
        def g(db): return 10 ** (db / 20.0)
        freq = np.fft.rfftfreq(len(data), 1.0 / sr)
        spec = np.fft.rfft(data, axis=0)
        mask = np.ones(len(freq), dtype=np.float32)
        mask[freq < 300]  *= g(lo)
        mask[(freq >= 300) & (freq < 3000)] *= g(mid)
        mask[freq >= 3000] *= g(hi)
        if data.ndim > 1:
            for ch in range(data.shape[1]): spec[:, ch] *= mask
        else:
            spec *= mask
        return np.fft.irfft(spec, n=len(data), axis=0).astype(np.float32)

    @staticmethod
    def _reverb(data: np.ndarray, wet: float) -> np.ndarray:
        if wet == 0.0: return data
        out = data.copy(); decay = 0.45
        for delay_ms in [29, 43, 71, 97]:
            delay = int(SR * delay_ms / 1000)
            if len(out) > delay:
                out[delay:] += data[: len(out) - delay] * decay * wet
            decay *= 0.55
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _compress(data: np.ndarray) -> np.ndarray:
        thr = 0.5; ratio = 4.0
        peak = np.abs(data)
        gain = np.where(peak > thr, thr + (peak - thr) / ratio, peak)
        safe = np.where(peak > 1e-9, gain / peak, 1.0)
        return (data * safe).astype(np.float32)

    # ── mix builder ──
    def _render_track_clip(self, clip: Clip, trk: Track,
                           from_beat: float, total_frames: int) -> Optional[np.ndarray]:
        s = clip.sample
        if s.data is None: return None
        spb    = SR / self.proj.bps()
        clip_end = clip.start_beat + clip.length_beats
        if clip_end <= from_beat: return None
        mix_off  = max(0, int((clip.start_beat - from_beat) * spb))
        samp_off = max(0, int((from_beat - clip.start_beat) * spb)) if clip.start_beat < from_beat else 0
        n = min(len(s.data) - samp_off, int(clip.length_beats * spb) - samp_off)
        if n <= 0: return None
        src = s.data[samp_off: samp_off + n].copy()
        # stereo normalise
        if src.ndim == 1 or src.shape[1] == 1:
            src = np.column_stack([src.ravel(), src.ravel()])
        elif src.shape[1] > 2:
            src = src[:, :2]
        # gain & fades
        src *= clip.gain
        if clip.fade_in > 0:
            fi = int(clip.fade_in * spb)
            fade = np.linspace(0, 1, min(fi, len(src)))
            src[:len(fade)] *= fade[:, None]
        if clip.fade_out > 0:
            fo = int(clip.fade_out * spb)
            fade = np.linspace(1, 0, min(fo, len(src)))
            src[-len(fade):] *= fade[:, None]
        # FX
        if trk.fx_eq_low or trk.fx_eq_mid or trk.fx_eq_hi:
            src = self._eq(src, s.sr, trk.fx_eq_low, trk.fx_eq_mid, trk.fx_eq_hi)
        if trk.fx_reverb > 0:
            src = self._reverb(src, trk.fx_reverb)
        if trk.fx_comp:
            src = self._compress(src)
        # pan
        if trk.pan != 0.0:
            src[:, 0] *= min(1.0, 1.0 - trk.pan)
            src[:, 1] *= min(1.0, 1.0 + trk.pan)
        # volume automation
        vol = trk.volume
        buf = np.zeros((total_frames, 2), dtype=np.float32)
        end = mix_off + len(src)
        if end > total_frames: src = src[:total_frames - mix_off]
        if len(src) > 0:
            buf[mix_off: mix_off + len(src)] = src * vol
        return buf

    def _build_beat_mix(self, from_beat: float, total_frames: int) -> np.ndarray:
        mix = np.zeros((total_frames, 2), dtype=np.float32)
        spb_f = SR / self.proj.bps()
        bar_beats = self.proj.ts_num
        step_beats = bar_beats / BEAT_STEPS
        max_b = self.proj.max_beat()
        for trk in self.proj.tracks.values():
            if trk.muted or trk.step_sample is None: continue
            s = trk.step_sample
            if s.data is None: continue
            sdata = s.data
            if sdata.ndim == 1: sdata = np.column_stack([sdata, sdata])
            elif sdata.shape[1] == 1: sdata = np.column_stack([sdata[:,0], sdata[:,0]])
            elif sdata.shape[1] > 2: sdata = sdata[:, :2]
            bar = int(from_beat / bar_beats)
            while bar * bar_beats < max_b:
                for i, on in enumerate(trk.steps):
                    if not on: continue
                    sb = bar * bar_beats + i * step_beats
                    if sb < from_beat: continue
                    moff = int((sb - from_beat) * spb_f)
                    if moff >= total_frames: break
                    vel = trk.step_vel[i] / 127.0 * trk.volume
                    n = min(len(sdata), total_frames - moff)
                    mix[moff: moff + n] += sdata[:n] * vel
                bar += 1
        return mix

    def _build_mix(self) -> np.ndarray:
        from_beat    = self.position
        total_frames = int((self.proj.max_beat() - from_beat) * SR / self.proj.bps()) + SR * 2
        mix = np.zeros((max(1, total_frames), 2), dtype=np.float32)
        solo_on = any(t.solo for t in self.proj.tracks.values())
        for clip in self.proj.clips.values():
            trk = self.proj.tracks.get(clip.track_id)
            if not trk or trk.muted: continue
            if solo_on and not trk.solo: continue
            buf = self._render_track_clip(clip, trk, from_beat, total_frames)
            if buf is not None: mix += buf
        mix += self._build_beat_mix(from_beat, total_frames)
        peak = np.max(np.abs(mix))
        if peak > 1.0: mix /= peak
        return mix

    def _start_stream(self):
        self._stop_stream()
        if not AUDIO_OK: return
        mix = self._build_mix()
        with self._lock:
            self._mix = mix; self._buf_pos = 0
        def cb(outdata, frames, t, status):
            with self._lock:
                if self._mix is None: outdata[:] = 0; return
                p = self._buf_pos; chunk = self._mix[p: p + frames]; n = len(chunk)
                if n < frames: outdata[:n] = chunk; outdata[n:] = 0
                else: outdata[:] = chunk
                self._buf_pos += frames
        try:
            self._stream = sd.OutputStream(
                samplerate=SR, channels=2, dtype="float32",
                blocksize=512, callback=cb)
            self._stream.start()
        except Exception as e:
            print(f"[STREAM] {e}")

    def _stop_stream(self):
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except: pass
            self._stream = None

    def play(self):
        if self.playing: return
        self.playing = True
        self._beat0 = self.position; self._wall = time.perf_counter()
        self._start_stream(); self._timer.start()

    def pause(self):
        if not self.playing: return
        self.playing = False; self._stop_stream(); self._timer.stop()

    def stop(self):
        self.pause(); self.position = 0.0

    def seek(self, beat: float):
        was = self.playing
        if was: self.pause()
        self.position = max(0.0, beat)
        if was: self.play()

    def rebuild(self):
        if self.playing:
            self._beat0 = self.position; self._wall = time.perf_counter()
            self._start_stream()

    def render_full(self, progress_cb=None) -> np.ndarray:
        total = int(self.proj.max_beat() * SR / self.proj.bps()) + SR
        mix   = np.zeros((total, 2), dtype=np.float32)
        solo_on = any(t.solo for t in self.proj.tracks.values())
        clips   = list(self.proj.clips.values())
        for i, clip in enumerate(clips):
            trk = self.proj.tracks.get(clip.track_id)
            if not trk or trk.muted: continue
            if solo_on and not trk.solo: continue
            buf = self._render_track_clip(clip, trk, 0.0, total)
            if buf is not None: mix += buf
            if progress_cb: progress_cb(int((i + 1) / max(len(clips), 1) * 75))
        mix += self._build_beat_mix(0.0, total)
        peak = np.max(np.abs(mix))
        if peak > 1.0: mix /= peak
        if progress_cb: progress_cb(100)
        return mix


# ══════════════════════════════════════════════════════════
#  STYLE HELPERS
# ══════════════════════════════════════════════════════════
_BTN_SS = (
    "QPushButton{{background:#1e1e2a;border:1px solid #2a2a3a;border-radius:4px;"
    "color:{fg};font-size:11px;font-weight:700;padding:0 10px;height:{h}px;letter-spacing:0.8px;}}"
    "QPushButton:hover{{background:#25253a;border-color:{hv};color:{hv};}}"
    "QPushButton:checked{{background:{cb};border-color:{hv};color:{hv};}}"
    "QPushButton:disabled{{color:#333344;border-color:#222230;}}"
)
_SL_SS = (
    "QSlider::groove:horizontal{{background:#2a2a3a;height:4px;border-radius:2px;}}"
    "QSlider::handle:horizontal{{background:{c};width:12px;height:12px;border-radius:6px;margin:-4px 0;}}"
    "QSlider::sub-page:horizontal{{background:{c};border-radius:2px;}}"
    "QSlider::groove:vertical{{background:#2a2a3a;width:4px;border-radius:2px;}}"
    "QSlider::handle:vertical{{background:{c};width:12px;height:12px;border-radius:6px;margin:0 -4px;}}"
    "QSlider::sub-page:vertical{{background:{c};border-radius:2px;}}"
)

def mk_btn(text, fg=None, hover="#00D4FF", chk_bg="#0e1a28",
           w=None, h=28, checkable=False) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(h)
    if w: b.setFixedWidth(w)
    b.setCheckable(checkable)
    b.setStyleSheet(_BTN_SS.format(fg=fg or TEXT.name(), hv=hover, cb=chk_bg, h=h))
    return b

def mk_slider(col=None, orient=Qt.Orientation.Horizontal,
              lo=0, hi=100, val=100, w=None, h=None) -> QSlider:
    s = QSlider(orient)
    s.setRange(lo, hi); s.setValue(val)
    if w: s.setFixedWidth(w)
    if h: s.setFixedHeight(h)
    s.setStyleSheet(_SL_SS.format(c=col or ACCENT.name()))
    return s

def menu_ss():
    return (f"QMenu{{background:{PANEL2.name()};border:1px solid {BORDER.name()};"
            f"color:{TEXT.name()};font-size:12px;padding:4px;}}"
            f"QMenu::item:selected{{background:#25253a;border-radius:3px;}}"
            f"QMenu::separator{{background:{BORDER.name()};height:1px;margin:4px 8px;}}")


# ══════════════════════════════════════════════════════════
#  TIMELINE RULER
# ══════════════════════════════════════════════════════════
class Ruler(QWidget):
    seek_signal = pyqtSignal(float)

    def __init__(self, proj: Project):
        super().__init__()
        self.proj = proj; self.bpx = DEFAULT_BPX
        self.view_off = 0.0; self.playhead = 0.0
        self.setFixedHeight(RULER_H); self.setMouseTracking(True)

    def b2x(self, b): return (b - self.view_off) * self.bpx
    def x2b(self, x): return x / self.bpx + self.view_off
    def set_playhead(self, b): self.playhead = b; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#0e0e16"))
        W = self.width(); total = W / self.bpx + self.view_off
        step = 1 if self.bpx >= 60 else (2 if self.bpx >= 30 else 4)
        p.setFont(QFont("Courier New", 8))
        b = math.floor(self.view_off / step) * step
        while b <= total + step:
            x = int(self.b2x(b)); bar = (b % 4 == 0)
            p.setPen(QPen(QColor("#888" if bar else "#3a3a4a")))
            p.drawLine(x, 6 if bar else 14, x, RULER_H)
            if bar:
                p.setPen(QPen(QColor("#aaaaaa")))
                p.drawText(x + 3, RULER_H - 4, str(int(b / 4) + 1))
            b += step
        px = int(self.b2x(self.playhead))
        p.setPen(QPen(ACCENT3, 1)); p.drawLine(px, 0, px, RULER_H)
        tri = QPainterPath()
        tri.moveTo(px - 5, 0); tri.lineTo(px + 5, 0); tri.lineTo(px, 8); tri.closeSubpath()
        p.fillPath(tri, ACCENT3); p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.seek_signal.emit(max(0.0, self.x2b(e.position().x())))


# ══════════════════════════════════════════════════════════
#  TRACK LANE
# ══════════════════════════════════════════════════════════
class TrackLane(QWidget):
    clip_moved    = pyqtSignal(int, int, float)
    clip_deleted  = pyqtSignal(int)
    clip_dropped  = pyqtSignal(int, int, float)
    clip_resized  = pyqtSignal(int, float)
    clip_split    = pyqtSignal(int, float)
    clip_selected = pyqtSignal(int)
    clip_copy_req = pyqtSignal(int)
    clip_cut_req  = pyqtSignal(int)
    RESIZE_ZONE   = 8

    def __init__(self, track: Track, proj: Project):
        super().__init__()
        self.track = track; self.proj = proj
        self.bpx = DEFAULT_BPX; self.view_off = 0.0; self.playhead = 0.0
        self.active_tool = "select"; self.selected_cid = None; self._cut_x = -1
        self._drag_cid = None; self._drag_off = 0.0
        self._resize_cid = None; self._resize_sx = 0; self._resize_sl = 0.0
        self.setAcceptDrops(True); self.setMouseTracking(True)
        self.setFixedHeight(TRACK_H)

    def b2x(self, b): return (b - self.view_off) * self.bpx
    def x2b(self, x): return x / self.bpx + self.view_off

    def clip_rect(self, c: Clip) -> QRectF:
        return QRectF(self.b2x(c.start_beat), 4,
                      max(4.0, c.length_beats * self.bpx), TRACK_H - 8)

    def clip_at(self, pos: QPointF) -> Optional[Clip]:
        for cid in reversed(self.track.clips):
            c = self.proj.clips.get(cid)
            if c and self.clip_rect(c).contains(pos): return c
        return None

    def in_resize(self, c: Clip, pos: QPointF) -> bool:
        return self.clip_rect(c).right() - pos.x() < self.RESIZE_ZONE

    # ── paint ──────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0d0d14"))
        # grid
        total = W / self.bpx + self.view_off; b = math.floor(self.view_off)
        while b <= total + 1:
            x = int(self.b2x(b))
            p.setPen(QPen(QColor("#222235" if b % 4 == 0 else "#1a1a28")))
            p.drawLine(x, 0, x, H); b += 1
        # clips
        for cid in self.track.clips:
            c = self.proj.clips.get(cid)
            if c: self._draw_clip(p, c)
        # playhead
        px = int(self.b2x(self.playhead))
        p.setPen(QPen(ACCENT3, 1)); p.drawLine(px, 0, px, H)
        # cut marker
        if self.active_tool == "cut" and self._cut_x >= 0:
            p.setPen(QPen(ACCENT3, 2)); p.drawLine(self._cut_x, 0, self._cut_x, H)
            p.setFont(QFont("Arial", 10)); p.setPen(QPen(ACCENT3))
            p.drawText(self._cut_x + 3, 14, "✂")
        p.end()

    def _draw_clip(self, p: QPainter, clip: Clip):
        r = self.clip_rect(clip)
        if r.right() < 0 or r.left() > self.width(): return
        col = QColor(self.track.color)
        fill = QColor(col); fill.setAlpha(48)
        bord = QColor(col); bord.setAlpha(180)
        p.setBrush(QBrush(fill)); p.setPen(QPen(bord, 1))
        p.drawRoundedRect(r, 3, 3)
        # waveform
        s = clip.sample
        if s and s.data is not None:
            w = max(1, int(r.width())); peaks = get_peaks(s, w)
            mid = r.top() + r.height() / 2; half = r.height() / 2 - 5
            wc = QColor(col); wc.setAlpha(180); p.setPen(QPen(wc, 1))
            for i in range(min(w, int(r.width()))):
                xi = int(r.left()) + i
                if xi > self.width(): break
                mn, mx = peaks[i]; y1 = mid + mn * half; y2 = mid + mx * half
                if y2 - y1 < 1: y2 = y1 + 1
                p.drawLine(xi, int(y1), xi, int(y2))
        # fade overlays
        if clip.fade_in > 0:
            fi = clip.fade_in * self.bpx
            g = QLinearGradient(r.left(), 0, r.left() + fi, 0)
            g.setColorAt(0, QColor(0, 0, 0, 130)); g.setColorAt(1, QColor(0, 0, 0, 0))
            p.fillRect(QRectF(r.left(), r.top(), fi, r.height()), QBrush(g))
        if clip.fade_out > 0:
            fo = clip.fade_out * self.bpx
            g = QLinearGradient(r.right() - fo, 0, r.right(), 0)
            g.setColorAt(0, QColor(0, 0, 0, 0)); g.setColorAt(1, QColor(0, 0, 0, 130))
            p.fillRect(QRectF(r.right() - fo, r.top(), fo, r.height()), QBrush(g))
        # label
        p.setPen(QPen(QColor(255, 255, 255, 170)))
        p.setFont(QFont("Courier New", 8))
        p.drawText(QRectF(r.left() + 4, r.top() + 3, r.width() - 8, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   (s.name if s else "?")[:40])
        # selection outline
        if clip.id == self.selected_cid:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt.PenStyle.DashLine))
            p.drawRoundedRect(r, 3, 3)
        # resize handle
        p.setBrush(QBrush(QColor(255, 255, 255, 22)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(r.right() - 6, r.top(), 6, r.height()))

    # ── mouse ──────────────────────────────────────────────
    def mousePressEvent(self, e):
        pos = e.position(); clip = self.clip_at(pos)

        if e.button() == Qt.MouseButton.RightButton and clip:
            menu = QMenu(self); menu.setStyleSheet(menu_ss())
            cp = menu.addAction("⎘  Copy Clip           Ctrl+C")
            cx = menu.addAction("✂⎘  Cut Clip             Ctrl+X")
            menu.addSeparator()
            sp = menu.addAction("✂  Split Here")
            menu.addSeparator()
            fi = menu.addAction("⟿  Set Fade In…")
            fo = menu.addAction("⟾  Set Fade Out…")
            gn = menu.addAction("🔊  Set Gain…")
            menu.addSeparator()
            dl = menu.addAction("🗑  Delete Clip          Del")
            act = menu.exec(e.globalPosition().toPoint())
            if act == dl: self.clip_deleted.emit(clip.id)
            elif act == sp: self.clip_split.emit(clip.id, self.x2b(pos.x()))
            elif act == cp:
                self.selected_cid = clip.id; self.clip_copy_req.emit(clip.id); self.update()
            elif act == cx:
                self.selected_cid = clip.id; self.clip_cut_req.emit(clip.id); self.update()
            elif act == fi:
                v, ok = QInputDialog.getDouble(self, "Fade In", "Beats:", clip.fade_in, 0, 16, 2)
                if ok: clip.fade_in = v; self.update()
            elif act == fo:
                v, ok = QInputDialog.getDouble(self, "Fade Out", "Beats:", clip.fade_out, 0, 16, 2)
                if ok: clip.fade_out = v; self.update()
            elif act == gn:
                v, ok = QInputDialog.getDouble(self, "Gain", "0.0 – 2.0:", clip.gain, 0, 2, 2)
                if ok: clip.gain = v; self.update()
            return

        if e.button() == Qt.MouseButton.LeftButton:
            if self.active_tool == "cut":
                if clip: self.clip_split.emit(clip.id, self.x2b(pos.x()))
                return
            if clip:
                self.selected_cid = clip.id; self.clip_selected.emit(clip.id); self.update()
            else:
                self.selected_cid = None; self.update(); return
            if self.in_resize(clip, pos):
                self._resize_cid = clip.id
                self._resize_sx = int(pos.x()); self._resize_sl = clip.length_beats
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self._drag_cid = clip.id
                self._drag_off = self.x2b(pos.x()) - clip.start_beat
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._resize_cid is not None:
            dx = pos.x() - self._resize_sx
            self.clip_resized.emit(self._resize_cid,
                                   max(0.25, self._resize_sl + dx / self.bpx))
            self.update(); return
        if self._drag_cid is not None:
            gp = e.globalPosition().toPoint()
            for ln in self.window().findChildren(TrackLane):
                tl = ln.mapToGlobal(QPoint(0, 0))
                if QRect(tl.x(), tl.y(), ln.width(), ln.height()).contains(gp):
                    nb = max(0.0, round((ln.x2b(gp.x() - tl.x()) - self._drag_off) * 4) / 4)
                    self.clip_moved.emit(self._drag_cid, ln.track.id, nb)
                    break
            return
        if self.active_tool == "cut":
            beat = self.x2b(pos.x()); clip = self.clip_at(pos)
            self._cut_x = (int(pos.x())
                           if clip and clip.start_beat + 0.1 < beat < clip.start_beat + clip.length_beats - 0.1
                           else -1)
            self.update(); return
        clip = self.clip_at(pos)
        if clip:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self.in_resize(clip, pos)
                           else Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor if self.active_tool == "cut"
                           else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag_cid = self._resize_cid = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, e): self._cut_x = -1; self.update()

    # ── drag & drop from library ────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasText() and e.mimeData().text().startswith("sample:"):
            e.acceptProposedAction()

    def dragMoveEvent(self, e): e.acceptProposedAction()

    def dropEvent(self, e):
        t = e.mimeData().text()
        if t.startswith("sample:"):
            sid  = int(t.split(":")[1])
            beat = max(0.0, round(self.x2b(e.position().x()) * 2) / 2)
            self.clip_dropped.emit(sid, self.track.id, beat)
            e.acceptProposedAction()

    def set_view(self, bpx, off): self.bpx = bpx; self.view_off = off; self.update()
    def set_playhead(self, b): self.playhead = b; self.update()
    def set_tool(self, t): self.active_tool = t; self._cut_x = -1; self.update()


# ══════════════════════════════════════════════════════════
#  AUTOMATION LANE
# ══════════════════════════════════════════════════════════
class AutoLane(QWidget):
    changed = pyqtSignal()

    def __init__(self, track: Track, proj: Project):
        super().__init__()
        self.track = track; self.proj = proj
        self.bpx = DEFAULT_BPX; self.view_off = 0.0
        self._drag_pt = None
        self.setFixedHeight(38); self.setMouseTracking(True)

    def b2x(self, b): return (b - self.view_off) * self.bpx
    def x2b(self, x): return x / self.bpx + self.view_off
    def y2v(self, y): return max(0.0, min(1.0, 1.0 - y / self.height()))
    def v2y(self, v): return int((1.0 - v) * self.height())

    def paintEvent(self, _):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#080812"))
        pts = sorted(self.track.vol_auto, key=lambda pt: pt.beat)
        if not pts:
            p.setPen(QPen(TEXTDIM, 1, Qt.PenStyle.DashLine))
            p.drawLine(0, self.height() // 2, self.width(), self.height() // 2)
            p.end(); return
        col = QColor(self.track.color)
        p.setPen(QPen(col, 1.5)); prev = None
        for pt in pts:
            x = int(self.b2x(pt.beat)); y = self.v2y(pt.value)
            if prev: p.drawLine(int(self.b2x(prev.beat)), self.v2y(prev.value), x, y)
            p.setBrush(QBrush(col)); p.drawEllipse(QPoint(x, y), 4, 4)
            prev = pt
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            beat = self.x2b(e.position().x()); val = self.y2v(e.position().y())
            for pt in self.track.vol_auto:
                if abs(self.b2x(pt.beat) - e.position().x()) < 10:
                    self._drag_pt = pt; self.update(); return
            new_pt = AutoPoint(beat, val)
            self.track.vol_auto.append(new_pt); self._drag_pt = new_pt
            self.update(); self.changed.emit()
        elif e.button() == Qt.MouseButton.RightButton:
            for pt in self.track.vol_auto:
                if abs(self.b2x(pt.beat) - e.position().x()) < 10:
                    self.track.vol_auto.remove(pt); self.update(); self.changed.emit(); break

    def mouseMoveEvent(self, e):
        if self._drag_pt:
            self._drag_pt.beat  = max(0.0, self.x2b(e.position().x()))
            self._drag_pt.value = self.y2v(e.position().y())
            self.update(); self.changed.emit()

    def mouseReleaseEvent(self, e): self._drag_pt = None
    def set_view(self, bpx, off): self.bpx = bpx; self.view_off = off; self.update()


# ══════════════════════════════════════════════════════════
#  TRACK HEADER
# ══════════════════════════════════════════════════════════
class TrackHeader(QWidget):
    mute_changed   = pyqtSignal(int, bool)
    solo_changed   = pyqtSignal(int, bool)
    volume_changed = pyqtSignal(int, float)
    pan_changed    = pyqtSignal(int, float)
    delete_clicked = pyqtSignal(int)
    rename_clicked = pyqtSignal(int)
    fx_clicked     = pyqtSignal(int)
    auto_clicked   = pyqtSignal(int)

    def __init__(self, track: Track):
        super().__init__()
        self.track = track
        self.setFixedSize(HDR_W, TRACK_H)
        self._build()

    def _build(self):
        col = self.track.color
        lo = QHBoxLayout(self); lo.setContentsMargins(5, 5, 5, 5); lo.setSpacing(4)
        bar = QFrame(); bar.setFixedWidth(3)
        bar.setStyleSheet(f"background:{col};border-radius:1px;"); lo.addWidget(bar)

        left = QVBoxLayout(); left.setSpacing(2)
        self.name_btn = QPushButton(self.track.name[:14])
        self.name_btn.setStyleSheet(
            f"QPushButton{{background:none;border:none;color:{TEXT.name()};"
            f"font-size:11px;font-weight:700;text-align:left;padding:0;}}"
            f"QPushButton:hover{{color:{ACCENT2.name()};}}")
        self.name_btn.clicked.connect(lambda: self.rename_clicked.emit(self.track.id))
        left.addWidget(self.name_btn)

        btns = QHBoxLayout(); btns.setSpacing(2)
        def mini(txt, c1, attr):
            b = QPushButton(txt); b.setFixedSize(22, 18); b.setCheckable(True)
            b.setChecked(getattr(self.track, attr))
            b.setStyleSheet(
                f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
                f"border-radius:3px;color:{TEXTDIM.name()};font-size:8px;font-weight:700;}}"
                f"QPushButton:hover{{border-color:{c1};color:{c1};}}"
                f"QPushButton:checked{{border-color:{c1};color:{c1};}}")
            return b

        self.m_btn = mini("M", ACCENT3.name(), "muted")
        self.m_btn.toggled.connect(lambda v: self.mute_changed.emit(self.track.id, v))
        self.s_btn = mini("S", ACCENT.name(), "solo")
        self.s_btn.toggled.connect(lambda v: self.solo_changed.emit(self.track.id, v))

        fx_btn = QPushButton("FX"); fx_btn.setFixedSize(24, 18)
        fx_btn.setStyleSheet(
            f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
            f"border-radius:3px;color:{TEXTDIM.name()};font-size:8px;font-weight:700;}}"
            f"QPushButton:hover{{border-color:{ACCENT4.name()};color:{ACCENT4.name()};}}")
        fx_btn.clicked.connect(lambda: self.fx_clicked.emit(self.track.id))

        au_btn = QPushButton("AUTO"); au_btn.setFixedSize(32, 18)
        au_btn.setStyleSheet(
            f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
            f"border-radius:3px;color:{TEXTDIM.name()};font-size:7px;font-weight:700;}}"
            f"QPushButton:hover{{border-color:{ACCENT2.name()};color:{ACCENT2.name()};}}")
        au_btn.clicked.connect(lambda: self.auto_clicked.emit(self.track.id))

        for b in [self.m_btn, self.s_btn, fx_btn, au_btn]: btns.addWidget(b)
        left.addLayout(btns); lo.addLayout(left)

        # vol + pan sliders
        right = QVBoxLayout(); right.setSpacing(2)
        vrow = QHBoxLayout(); vrow.setSpacing(2)
        vrow.addWidget(QLabel("V"))
        self.vol_sl = mk_slider(col, lo=0, hi=100, val=int(self.track.volume * 100), w=56)
        self.vol_sl.valueChanged.connect(lambda v: self.volume_changed.emit(self.track.id, v / 100))
        vrow.addWidget(self.vol_sl)
        prow = QHBoxLayout(); prow.setSpacing(2)
        prow.addWidget(QLabel("P"))
        self.pan_sl = mk_slider(ACCENT2.name(), lo=-100, hi=100, val=int(self.track.pan * 100), w=56)
        self.pan_sl.valueChanged.connect(lambda v: self.pan_changed.emit(self.track.id, v / 100))
        prow.addWidget(self.pan_sl)
        for ql in self.findChildren(QLabel):
            ql.setStyleSheet(f"color:{TEXTDIM.name()};font-size:9px;")
        right.addLayout(vrow); right.addLayout(prow); lo.addLayout(right)

        xb = QPushButton("×"); xb.setFixedSize(16, 16)
        xb.setStyleSheet(
            f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
            f"border-radius:3px;color:#ff5555;font-size:13px;font-weight:bold;}}"
            f"QPushButton:hover{{border-color:#ff5555;background:#2a1a1a;}}")
        xb.clicked.connect(lambda: self.delete_clicked.emit(self.track.id))
        lo.addWidget(xb)

        self.setStyleSheet(
            f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};"
            f"border-right:1px solid {BORDER.name()};")


# ══════════════════════════════════════════════════════════
#  FX DIALOG
# ══════════════════════════════════════════════════════════
class FXDialog(QDialog):
    def __init__(self, track: Track, parent=None):
        super().__init__(parent)
        self.track = track
        self.setWindowTitle(f"FX — {track.name}"); self.setFixedSize(340, 300)
        self.setStyleSheet(
            f"QDialog{{background:{PANEL2.name()};color:{TEXT.name()};}}"
            f"QLabel{{color:{TEXT.name()};font-size:11px;}}"
            f"QGroupBox{{color:{TEXTDIM.name()};font-size:10px;border:1px solid {BORDER.name()};"
            f"border-radius:4px;margin-top:8px;padding-top:6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        lo = QVBoxLayout(self); lo.setSpacing(10); lo.setContentsMargins(16, 14, 16, 14)

        title = QLabel(f"⚡  FX Chain — {track.name}")
        title.setStyleSheet(f"color:{ACCENT2.name()};font-size:13px;font-weight:700;letter-spacing:1px;")
        lo.addWidget(title)

        # Reverb
        rg = QGroupBox("Reverb"); rl = QHBoxLayout(rg)
        rl.addWidget(QLabel("Wet"))
        self.rv_sl = mk_slider(ACCENT4.name(), lo=0, hi=100,
                                val=int(track.fx_reverb * 100), w=150)
        self.rv_lbl = QLabel(f"{int(track.fx_reverb*100)}%")
        self.rv_lbl.setStyleSheet(f"color:{ACCENT4.name()};font-family:monospace;min-width:34px;")
        self.rv_sl.valueChanged.connect(lambda v: (
            setattr(track, "fx_reverb", v / 100),
            self.rv_lbl.setText(f"{v}%")))
        rl.addWidget(self.rv_sl); rl.addWidget(self.rv_lbl); lo.addWidget(rg)

        # EQ
        eg = QGroupBox("3-Band EQ  (±12 dB)"); eg_lo = QGridLayout(eg)
        for col_i, (lbl, attr, col_c) in enumerate([
                ("Low",  "fx_eq_low", ACCENT.name()),
                ("Mid",  "fx_eq_mid", ACCENT2.name()),
                ("High", "fx_eq_hi",  ACCENT3.name())]):
            eg_lo.addWidget(QLabel(lbl), 0, col_i, Qt.AlignmentFlag.AlignCenter)
            sl = mk_slider(col_c, Qt.Orientation.Vertical,
                           lo=-120, hi=120, val=int(getattr(track, attr) * 10))
            sl.setFixedHeight(70)
            val_lbl = QLabel(f"{getattr(track, attr):.1f}")
            val_lbl.setStyleSheet(f"color:{col_c};font-size:9px;font-family:monospace;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sl.valueChanged.connect(lambda v, a=attr, l=val_lbl: (
                setattr(track, a, v / 10), l.setText(f"{v/10:.1f}")))
            eg_lo.addWidget(sl, 1, col_i, Qt.AlignmentFlag.AlignCenter)
            eg_lo.addWidget(val_lbl, 2, col_i, Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(eg)

        # Compressor
        cg = QGroupBox("Compressor"); cl = QHBoxLayout(cg)
        self.comp_chk = QCheckBox("Enable (4:1 ratio, −6 dB threshold)")
        self.comp_chk.setChecked(track.fx_comp)
        self.comp_chk.setStyleSheet(f"color:{TEXT.name()};")
        self.comp_chk.toggled.connect(lambda v: setattr(track, "fx_comp", v))
        cl.addWidget(self.comp_chk); lo.addWidget(cg)

        close = mk_btn("CLOSE", hover=ACCENT2.name(), w=80)
        close.clicked.connect(self.accept)
        lo.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


# ══════════════════════════════════════════════════════════
#  BEAT EDITOR
# ══════════════════════════════════════════════════════════
class BeatStepBtn(QPushButton):
    """Single step button with right-click velocity editing."""
    velocity_changed = pyqtSignal(int, int)   # step_index, velocity

    def __init__(self, track: Track, step_idx: int):
        super().__init__()
        self.track = track; self.idx = step_idx
        self.setFixedSize(28, TRACK_H - 18)
        self.setCheckable(True)
        self.setChecked(track.steps[step_idx])
        self._update_style()
        self.toggled.connect(self._on_toggle)

    def _update_style(self):
        col = QColor(self.track.color)
        vel = self.track.step_vel[self.idx]
        alpha = int(80 + vel / 127 * 175)
        active = col.name(); active_a = QColor(col); active_a.setAlpha(alpha)
        self.setStyleSheet(
            f"QPushButton{{background:{PANEL2.name()};border:1px solid {BORDER.name()};"
            f"border-radius:3px;}}"
            f"QPushButton:checked{{background:{active};"
            f"border-color:{active};box-shadow:0 0 6px {active};}}"
            f"QPushButton:hover{{border-color:{active};}}")

    def _on_toggle(self, v):
        self.track.steps[self.idx] = v

    def contextMenuEvent(self, e):
        menu = QMenu(self); menu.setStyleSheet(menu_ss())
        menu.addAction(f"Step {self.idx+1} — Velocity: {self.track.step_vel[self.idx]}")
        menu.addSeparator()
        for v in [25, 50, 75, 100, 127]:
            a = menu.addAction(f"Set velocity {v}")
            a.triggered.connect(lambda _, vel=v: self._set_vel(vel))
        menu.exec(e.globalPos())

    def _set_vel(self, v: int):
        self.track.step_vel[self.idx] = v
        self._update_style()
        self.velocity_changed.emit(self.idx, v)


class BeatRow(QWidget):
    def __init__(self, track: Track, proj: Project, engine: AudioEngine):
        super().__init__()
        self.track = track; self.proj = proj; self.engine = engine
        self.setFixedHeight(TRACK_H)
        lo = QHBoxLayout(self); lo.setContentsMargins(4, 4, 4, 4); lo.setSpacing(3)
        # color bar
        bar = QFrame(); bar.setFixedWidth(3)
        bar.setStyleSheet(f"background:{track.color};border-radius:1px;"); lo.addWidget(bar)
        # sample picker
        name = track.step_sample.name[:9] if track.step_sample else "— none —"
        self.smp_btn = QPushButton(name); self.smp_btn.setFixedWidth(84)
        col = QColor(track.color)
        self.smp_btn.setStyleSheet(
            f"QPushButton{{background:#1a1a28;border:1px solid {BORDER.name()};"
            f"border-radius:3px;color:{col.name()};font-size:10px;padding:0 4px;}}"
            f"QPushButton:hover{{border-color:{col.name()};}}")
        self.smp_btn.clicked.connect(self._pick_sample)
        lo.addWidget(self.smp_btn)
        # step buttons with bar separators
        self.step_btns = []
        for i in range(BEAT_STEPS):
            if i > 0 and i % 4 == 0:
                sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color:{BORDER.name()};"); lo.addWidget(sep)
            b = BeatStepBtn(track, i)
            b.toggled.connect(lambda _, t=self: t.engine.rebuild() if t.engine.playing else None)
            self.step_btns.append(b); lo.addWidget(b)
        lo.addStretch()
        self.setStyleSheet(f"background:{PANEL2.name()};border-bottom:1px solid {BORDER.name()};")

    def _pick_sample(self):
        if not self.proj.samples:
            QMessageBox.information(self, "No Samples", "Import audio files first."); return
        names = [s.name for s in self.proj.samples.values()]
        ids   = list(self.proj.samples.keys())
        name, ok = QInputDialog.getItem(self, "Assign Sample",
                                        f"Sample for beat track '{self.track.name}':",
                                        names, 0, False)
        if ok:
            idx = names.index(name)
            self.track.step_sample = self.proj.samples[ids[idx]]
            self.smp_btn.setText(name[:9])
            if self.engine.playing: self.engine.rebuild()


class BeatEditor(QWidget):
    def __init__(self, proj: Project, engine: AudioEngine):
        super().__init__()
        self.proj = proj; self.engine = engine
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        # toolbar
        tb = QWidget(); tb.setFixedHeight(32)
        tb.setStyleSheet(f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(10, 4, 10, 4); tbl.setSpacing(8)
        tbl.addWidget(QLabel("🥁  BEAT EDITOR"))
        tbl.addStretch()
        clr = mk_btn("Clear All", hover=ACCENT3.name(), w=80, h=24)
        clr.clicked.connect(self._clear_all); tbl.addWidget(clr)
        lo.addWidget(tb)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            f"QScrollArea{{background:{PANEL.name()};border:none;}}"
            f"QScrollBar:vertical{{background:{PANEL.name()};width:6px;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER.name()};border-radius:3px;}}")
        self.inner = QWidget(); self.inner.setStyleSheet(f"background:{PANEL.name()};")
        self.rows_lo = QVBoxLayout(self.inner)
        self.rows_lo.setContentsMargins(0, 0, 0, 0); self.rows_lo.setSpacing(0)
        self.rows_lo.addStretch()
        self.scroll.setWidget(self.inner); lo.addWidget(self.scroll)
        self._rows: Dict[int, BeatRow] = {}

    def rebuild(self):
        for w in list(self._rows.values()):
            self.rows_lo.removeWidget(w); w.setParent(None)
        self._rows.clear()
        for tid, trk in self.proj.tracks.items():
            row = BeatRow(trk, self.proj, self.engine)
            self.rows_lo.insertWidget(self.rows_lo.count() - 1, row)
            self._rows[tid] = row

    def _clear_all(self):
        for trk in self.proj.tracks.values():
            trk.steps = [False] * BEAT_STEPS
        self.rebuild()
        if self.engine.playing: self.engine.rebuild()


# ══════════════════════════════════════════════════════════
#  PIANO ROLL
# ══════════════════════════════════════════════════════════
class PianoCanvas(QWidget):
    """The scrollable grid inside the Piano Roll."""
    changed = pyqtSignal()

    def __init__(self, roll: "PianoRoll"):
        super().__init__()
        self.roll = roll
        self._drag_note: Optional[MidiNote] = None
        self._drag_sx = 0; self._drag_sbeat = 0.0; self._drag_spitch = 0
        self._resize_note: Optional[MidiNote] = None
        self._resize_sx = 0; self._resize_sl = 0.0
        self.setMouseTracking(True)
        total_h = TOTAL_NOTES * NOTE_H
        self.setMinimumSize(2000, total_h)
        self.setMaximumHeight(total_h)

    def b2x(self, b): return PIANO_W + (b - self.roll.view_off) * self.roll.bpx
    def x2b(self, x): return max(0.0, (x - PIANO_W) / self.roll.bpx + self.roll.view_off)
    def pitch_at(self, y): return TOTAL_NOTES - 1 - int(y / NOTE_H)

    def note_rect(self, n: MidiNote) -> QRectF:
        return QRectF(self.b2x(n.start), (TOTAL_NOTES - 1 - n.pitch) * NOTE_H,
                      max(4.0, n.length * self.roll.bpx), NOTE_H - 1)

    def note_at(self, pos: QPointF) -> Optional[MidiNote]:
        t = self.roll.track
        if not t: return None
        for n in reversed(t.midi_notes):
            if self.note_rect(n).contains(pos): return n
        return None

    def in_resize(self, n: MidiNote, pos: QPointF) -> bool:
        return self.note_rect(n).right() - pos.x() < 8

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0c0c14"))
        # piano keys
        for i in range(TOTAL_NOTES):
            pitch = TOTAL_NOTES - 1 - i; note = pitch % 12; y = i * NOTE_H
            is_black = note in (1, 3, 6, 8, 10)
            p.fillRect(0, y, PIANO_W, NOTE_H - 1,
                       QColor("#101020") if is_black else QColor("#1c1c30"))
            if not is_black and note == 0:
                p.setPen(QPen(TEXTDIM)); p.setFont(QFont("Courier New", 7))
                p.drawText(2, y + NOTE_H - 2, f"C{pitch // 12}")
        p.setPen(QPen(BORDER, 1))
        for i in range(TOTAL_NOTES): p.drawLine(PIANO_W, i * NOTE_H, W, i * NOTE_H)
        # beat grid
        r = self.roll; total_b = (W - PIANO_W) / r.bpx + r.view_off
        b = math.floor(r.view_off); step = 1 if r.bpx >= 40 else 2
        while b <= total_b + step:
            x = int(self.b2x(b)); bar = (b % 4 == 0)
            p.setPen(QPen(QColor("#2a2a55" if bar else "#1a1a28")))
            p.drawLine(x, 0, x, H); b += step
        # notes
        t = self.roll.track
        if t:
            for n in t.midi_notes:
                r2 = self.note_rect(n)
                alpha = int(80 + n.velocity / 127 * 175)
                col = QColor(ACCENT2); col.setAlpha(alpha)
                p.fillRect(r2, col)
                p.setPen(QPen(ACCENT2.lighter(130), 1)); p.drawRect(r2)
                # velocity bar
                vbar_h = max(2, int(n.velocity / 127 * (NOTE_H - 3)))
                p.fillRect(QRectF(r2.left() + 1, r2.bottom() - vbar_h, 3, vbar_h),
                           QColor("#ffffff80"))
        p.end()

    def mousePressEvent(self, e):
        pos = e.position(); t = self.roll.track
        if not t: return
        note = self.note_at(pos)
        if e.button() == Qt.MouseButton.LeftButton:
            if self.roll.active_tool == "draw":
                if note:
                    if self.in_resize(note, pos):
                        self._resize_note = note
                        self._resize_sx = int(pos.x()); self._resize_sl = note.length
                    else:
                        self._drag_note = note
                        self._drag_sx = int(pos.x()); self._drag_sbeat = note.start
                        self._drag_spitch = note.pitch
                else:
                    beat = round(self.x2b(pos.x()) / self.roll.quant()) * self.roll.quant()
                    pitch = max(0, min(TOTAL_NOTES - 1, self.pitch_at(pos.y())))
                    n = MidiNote(pitch, beat, self.roll.quant(), 100)
                    t.midi_notes.append(n)
                    self._drag_note = n
                    self._drag_sx = int(pos.x()); self._drag_sbeat = n.start
                    self._drag_spitch = n.pitch
                    self.changed.emit()
            elif self.roll.active_tool == "erase":
                if note: t.midi_notes.remove(note); self.changed.emit()
        elif e.button() == Qt.MouseButton.RightButton and note:
            t.midi_notes.remove(note); self.changed.emit()
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._resize_note:
            dx = pos.x() - self._resize_sx
            self._resize_note.length = max(self.roll.quant(),
                                            self._resize_sl + dx / self.roll.bpx)
            self.update(); return
        if self._drag_note:
            db = self.x2b(pos.x()) - self.x2b(self._drag_sx)
            new_b = max(0.0, self._drag_sbeat + db)
            new_b = round(new_b / self.roll.quant()) * self.roll.quant()
            dp = self.pitch_at(pos.y()) - self.pitch_at(self._drag_spitch * NOTE_H + NOTE_H // 2)
            self._drag_note.start = new_b
            self._drag_note.pitch = max(0, min(TOTAL_NOTES - 1, self._drag_spitch + dp))
            self.update(); return
        note = self.note_at(pos)
        if note:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self.in_resize(note, pos)
                           else Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor if self.roll.active_tool == "draw"
                           else Qt.CursorShape.ForbiddenCursor)

    def mouseReleaseEvent(self, e):
        self._drag_note = None; self._resize_note = None
        self.setCursor(Qt.CursorShape.ArrowCursor)


class PianoRoll(QWidget):
    def __init__(self, proj: Project, engine: AudioEngine):
        super().__init__()
        self.proj = proj; self.engine = engine
        self.track: Optional[Track] = None
        self.active_tool = "draw"
        self.bpx = 60.0; self.view_off = 0.0
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        # toolbar
        tb = QWidget(); tb.setFixedHeight(34)
        tb.setStyleSheet(f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(10, 4, 10, 4); tbl.setSpacing(8)
        self.track_lbl = QLabel("No track selected")
        self.track_lbl.setStyleSheet(f"color:{ACCENT2.name()};font-size:11px;font-weight:700;")
        tbl.addWidget(self.track_lbl); tbl.addStretch()
        self.btn_draw  = mk_btn("✏  Draw",  hover=ACCENT.name(),  w=80, checkable=True, h=26)
        self.btn_erase = mk_btn("⌫  Erase", hover=ACCENT3.name(), w=80, checkable=True, h=26)
        self.btn_draw.setChecked(True)
        self.btn_draw.clicked.connect(lambda: self._set_tool("draw"))
        self.btn_erase.clicked.connect(lambda: self._set_tool("erase"))
        tbl.addWidget(self.btn_draw); tbl.addWidget(self.btn_erase)
        lbl_q = QLabel("Quantise:"); lbl_q.setStyleSheet(f"color:{TEXTDIM.name()};font-size:10px;")
        self.q_combo = QComboBox()
        self.q_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.q_combo.setCurrentIndex(2)
        self.q_combo.setStyleSheet(
            f"background:{PANEL2.name()};color:{TEXT.name()};"
            f"border:1px solid {BORDER.name()};border-radius:3px;padding:2px;")
        tbl.addWidget(lbl_q); tbl.addWidget(self.q_combo)
        lo.addWidget(tb)
        # scroll area for canvas
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(False)
        self.scroll.setStyleSheet(
            f"QScrollArea{{background:{PANEL.name()};border:none;}}"
            f"QScrollBar:vertical{{background:{PANEL.name()};width:8px;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER.name()};border-radius:4px;}}"
            f"QScrollBar:horizontal{{background:{PANEL.name()};height:8px;border:none;}}"
            f"QScrollBar::handle:horizontal{{background:{BORDER.name()};border-radius:4px;}}")
        self.canvas = PianoCanvas(self)
        self.canvas.changed.connect(lambda: None)
        self.scroll.setWidget(self.canvas); lo.addWidget(self.scroll)

    def quant(self) -> float:
        return {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125}.get(
            self.q_combo.currentText(), 0.25)

    def set_track(self, trk: Track):
        self.track = trk; self.track_lbl.setText(f"🎹  {trk.name}"); self.canvas.update()

    def _set_tool(self, t):
        self.active_tool = t
        self.btn_draw.setChecked(t == "draw")
        self.btn_erase.setChecked(t == "erase")


# ══════════════════════════════════════════════════════════
#  SAMPLE ITEM & LIBRARY
# ══════════════════════════════════════════════════════════
class SampleItem(QWidget):
    preview_clicked = pyqtSignal(int)

    def __init__(self, s: Sample):
        super().__init__()
        self.sample = s; self.setFixedHeight(50)
        lo = QHBoxLayout(self); lo.setContentsMargins(8, 4, 8, 4); lo.setSpacing(5)
        col = "#B44FFF" if s.ext == "MP3" else "#00D4FF"
        badge = QLabel(s.ext[:4]); badge.setFixedSize(26, 24)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background:#161630;border:1px solid {col};border-radius:3px;"
            f"color:{col};font-size:8px;font-weight:700;font-family:monospace;")
        lo.addWidget(badge)
        info = QVBoxLayout(); info.setSpacing(0)
        nl = QLabel(s.name); nl.setStyleSheet(f"color:{TEXT.name()};font-size:11px;font-weight:600;")
        nl.setMaximumWidth(110)
        dur = f"{int(s.duration//60)}:{int(s.duration%60):02d}"
        dl = QLabel(dur); dl.setStyleSheet(f"color:{TEXTDIM.name()};font-size:9px;font-family:monospace;")
        info.addWidget(nl); info.addWidget(dl); lo.addLayout(info); lo.addStretch()
        self.play_btn = QPushButton("▶"); self.play_btn.setFixedSize(22, 22)
        self.play_btn.setCheckable(True)
        self.play_btn.setStyleSheet(
            f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
            f"border-radius:3px;color:{TEXTDIM.name()};font-size:9px;}}"
            f"QPushButton:hover{{border-color:{ACCENT.name()};color:{ACCENT.name()};}}"
            f"QPushButton:checked{{border-color:{ACCENT3.name()};color:{ACCENT3.name()};background:#2e1a0a;}}")
        self.play_btn.clicked.connect(lambda: self.preview_clicked.emit(s.id))
        lo.addWidget(self.play_btn)
        self.setStyleSheet(
            f"background:{PANEL2.name()};border:1px solid {BORDER.name()};border-radius:4px;")
        self._dp = QPoint()

    def set_playing(self, v: bool):
        self.play_btn.setChecked(v); self.play_btn.setText("■" if v else "▶")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self._dp = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.MouseButton.LeftButton and
                (e.position().toPoint() - self._dp).manhattanLength() > 6):
            drag = QDrag(self); mime = QMimeData()
            mime.setText(f"sample:{self.sample.id}"); drag.setMimeData(mime)
            px = QPixmap(100, 18); px.fill(QColor(0, 245, 160, 50))
            pp = QPainter(px); pp.setPen(QPen(ACCENT))
            pp.setFont(QFont("Arial", 8))
            pp.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, self.sample.name[:18])
            pp.end(); drag.setPixmap(px); drag.setHotSpot(QPoint(50, 9))
            drag.exec(Qt.DropAction.CopyAction)
        super().mouseMoveEvent(e)


class SampleLibrary(QWidget):
    def __init__(self, proj: Project, engine: AudioEngine):
        super().__init__()
        self.proj = proj; self.engine = engine
        self._items: Dict[int, SampleItem] = {}; self._prev_sid = None
        self.setMinimumWidth(190); self.setMaximumWidth(240)
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        hdr = QLabel("  SAMPLES")
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};"
            f"color:{TEXTDIM.name()};font-size:10px;font-family:monospace;letter-spacing:2px;")
        lo.addWidget(hdr)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            f"QScrollArea{{background:{PANEL.name()};border:none;}}"
            f"QScrollBar:vertical{{background:{PANEL.name()};width:5px;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER.name()};border-radius:2px;}}")
        self.ct = QWidget(); self.ct.setStyleSheet(f"background:{PANEL.name()};")
        self.vb = QVBoxLayout(self.ct)
        self.vb.setContentsMargins(5, 5, 5, 5); self.vb.setSpacing(3)
        self.empty = QLabel("Import audio\nfiles to start")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet(f"color:{TEXTDIM.name()};font-size:10px;")
        self.vb.addWidget(self.empty); self.vb.addStretch()
        self.scroll.setWidget(self.ct); lo.addWidget(self.scroll)

    def add_sample(self, s: Sample):
        self.empty.hide()
        it = SampleItem(s); it.preview_clicked.connect(self._toggle_prev)
        self._items[s.id] = it
        self.vb.insertWidget(self.vb.count() - 1, it)

    def _toggle_prev(self, sid: int):
        if not AUDIO_OK: return
        if self._prev_sid is not None:
            if self._prev_sid in self._items: self._items[self._prev_sid].set_playing(False)
            try: sd.stop()
            except: pass
        if self._prev_sid == sid: self._prev_sid = None; return
        s = self.proj.samples.get(sid)
        if not s or s.data is None: return
        self._prev_sid = sid; self._items[sid].set_playing(True)
        d = s.data
        if d.ndim == 1: d = np.column_stack([d, d])
        elif d.shape[1] == 1: d = np.column_stack([d[:, 0], d[:, 0]])
        elif d.shape[1] > 2: d = d[:, :2]
        try: sd.play(d, s.sr)
        except: pass
        def _stop():
            time.sleep(s.duration + 0.2)
            if self._prev_sid == sid:
                self._prev_sid = None
                try: sd.stop()
                except: pass
                if sid in self._items: self._items[sid].set_playing(False)
        threading.Thread(target=_stop, daemon=True).start()


# ══════════════════════════════════════════════════════════
#  MIXER PANEL
# ══════════════════════════════════════════════════════════
class MixerStrip(QWidget):
    def __init__(self, track: Track):
        super().__init__()
        self.track = track; self.setFixedWidth(80)
        lo = QVBoxLayout(self); lo.setContentsMargins(6, 8, 6, 8); lo.setSpacing(4)
        col = QColor(track.color)
        ci = QFrame(); ci.setFixedHeight(4)
        ci.setStyleSheet(f"background:{col.name()};border-radius:2px;"); lo.addWidget(ci)
        nl = QLabel(track.name[:8]); nl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nl.setStyleSheet(f"color:{TEXT.name()};font-size:10px;font-weight:600;")
        nl.setWordWrap(True); lo.addWidget(nl)
        # VU-style fader
        self.fdr = mk_slider(col.name(), Qt.Orientation.Vertical,
                              lo=0, hi=150, val=int(track.volume * 100), h=110)
        self.fdr.valueChanged.connect(lambda v: setattr(track, "volume", v / 100))
        lo.addWidget(self.fdr, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.vol_lbl = QLabel(f"{int(track.volume*100)}%")
        self.vol_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vol_lbl.setStyleSheet(f"color:{TEXTDIM.name()};font-size:9px;font-family:monospace;")
        self.fdr.valueChanged.connect(lambda v: self.vol_lbl.setText(f"{v}%"))
        lo.addWidget(self.vol_lbl)
        # Pan
        pan = mk_slider(ACCENT2.name(), lo=-100, hi=100, val=int(track.pan * 100), w=68)
        pan.valueChanged.connect(lambda v: setattr(track, "pan", v / 100))
        lo.addWidget(pan)
        pan_lbl = QLabel("PAN"); pan_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pan_lbl.setStyleSheet(f"color:{TEXTDIM.name()};font-size:8px;font-family:monospace;")
        lo.addWidget(pan_lbl)
        # M / S
        ms = QHBoxLayout(); ms.setSpacing(2)
        def tc(txt, c, attr):
            b = QPushButton(txt); b.setFixedSize(30, 20); b.setCheckable(True)
            b.setChecked(getattr(track, attr))
            b.setStyleSheet(
                f"QPushButton{{background:none;border:1px solid {BORDER.name()};"
                f"border-radius:3px;color:{TEXTDIM.name()};font-size:9px;font-weight:700;}}"
                f"QPushButton:checked{{border-color:{c};color:{c};}}")
            b.toggled.connect(lambda v, t=track, a=attr: setattr(t, a, v))
            return b
        ms.addWidget(tc("M", ACCENT3.name(), "muted"))
        ms.addWidget(tc("S", ACCENT.name(),  "solo"))
        lo.addLayout(ms)
        self.setStyleSheet(
            f"background:{PANEL2.name()};border:1px solid {BORDER.name()};border-radius:6px;")


class MixerPanel(QWidget):
    def __init__(self, proj: Project, engine: AudioEngine):
        super().__init__()
        self.proj = proj; self.engine = engine
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        hdr = QLabel("  MIXER")
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};"
            f"color:{TEXTDIM.name()};font-size:10px;font-family:monospace;letter-spacing:2px;")
        lo.addWidget(hdr)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(f"QScrollArea{{background:{PANEL.name()};border:none;}}"
                                  f"QScrollBar:horizontal{{background:{PANEL.name()};height:8px;border:none;}}"
                                  f"QScrollBar::handle:horizontal{{background:{BORDER.name()};border-radius:4px;}}")
        self.inner = QWidget(); self.inner.setStyleSheet(f"background:{PANEL.name()};")
        self.ch_lo = QHBoxLayout(self.inner)
        self.ch_lo.setContentsMargins(8, 8, 8, 8); self.ch_lo.setSpacing(6)
        self.ch_lo.addStretch()
        self.scroll.setWidget(self.inner); lo.addWidget(self.scroll)
        self._strips: Dict[int, MixerStrip] = {}

    def rebuild(self):
        for w in list(self._strips.values()):
            self.ch_lo.removeWidget(w); w.setParent(None)
        self._strips.clear()
        for tid, trk in self.proj.tracks.items():
            strip = MixerStrip(trk)
            self.ch_lo.insertWidget(self.ch_lo.count() - 1, strip)
            self._strips[tid] = strip


# ══════════════════════════════════════════════════════════
#  ARRANGEMENT
# ══════════════════════════════════════════════════════════
class TrackContainer(QWidget):
    """One track row + optional automation lane."""
    def __init__(self, track: Track, proj: Project):
        super().__init__()
        self.track = track
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        row = QWidget(); row.setFixedHeight(TRACK_H)
        rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        self.header = TrackHeader(track)
        self.lane   = TrackLane(track, proj)
        rl.addWidget(self.header); rl.addWidget(self.lane)
        lo.addWidget(row)
        self.auto_lane = AutoLane(track, proj)
        self.auto_lane.setVisible(False)
        lo.addWidget(self.auto_lane)
        self.setStyleSheet(f"border-bottom:1px solid {BORDER.name()};")


class Arrangement(QWidget):
    track_deleted = pyqtSignal(int)
    add_track_req = pyqtSignal()
    seek_signal   = pyqtSignal(float)
    open_piano    = pyqtSignal(int)      # track_id

    def __init__(self, proj: Project):
        super().__init__()
        self.proj = proj; self.bpx = DEFAULT_BPX; self.voff = 0.0
        self._containers: Dict[int, TrackContainer] = {}
        self._clipboard: Optional[Clip] = None
        self._selected_cid: Optional[int] = None
        self._show_auto = False

        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        # ruler row
        rr = QWidget(); rr.setFixedHeight(RULER_H)
        rl = QHBoxLayout(rr); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        sp = QWidget(); sp.setFixedSize(HDR_W, RULER_H)
        sp.setStyleSheet(
            f"background:{PANEL.name()};border-bottom:1px solid {BORDER.name()};"
            f"border-right:1px solid {BORDER.name()};")
        rl.addWidget(sp)
        self.ruler = Ruler(proj); self.ruler.seek_signal.connect(self.seek_signal)
        rl.addWidget(self.ruler); lo.addWidget(rr)
        # tracks scroll
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{BG.name()};}}"
            f"QScrollBar:vertical{{background:{PANEL.name()};width:8px;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER.name()};border-radius:4px;min-height:20px;}}")
        self.tw = QWidget(); self.tw.setStyleSheet(f"background:{BG.name()};")
        self.tvb = QVBoxLayout(self.tw)
        self.tvb.setContentsMargins(0, 0, 0, 0); self.tvb.setSpacing(0)
        self.add_btn = QPushButton("＋  ADD TRACK"); self.add_btn.setFixedHeight(34)
        self.add_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"border-top:1px dashed {BORDER.name()};color:{TEXTDIM.name()};"
            f"font-size:12px;font-weight:700;letter-spacing:1px;}}"
            f"QPushButton:hover{{color:{ACCENT2.name()};background:rgba(0,212,255,0.04);}}")
        self.add_btn.clicked.connect(self.add_track_req)
        self.tvb.addWidget(self.add_btn); self.tvb.addStretch()
        self.scroll.setWidget(self.tw); lo.addWidget(self.scroll)
        # hscroll
        hr = QWidget(); hr.setFixedHeight(14)
        hrl = QHBoxLayout(hr); hrl.setContentsMargins(0, 0, 0, 0); hrl.setSpacing(0)
        pad = QWidget(); pad.setFixedWidth(HDR_W)
        pad.setStyleSheet(f"background:{PANEL.name()};border-top:1px solid {BORDER.name()};")
        hrl.addWidget(pad)
        self.hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self.hscroll.setRange(0, 3000)
        self.hscroll.setStyleSheet(
            f"QScrollBar:horizontal{{background:{PANEL.name()};height:14px;"
            f"border-top:1px solid {BORDER.name()};}}"
            f"QScrollBar::handle:horizontal{{background:{BORDER.name()};border-radius:3px;min-width:30px;}}"
            f"QScrollBar::handle:horizontal:hover{{background:#444;}}"
            f"QScrollBar::add-line,QScrollBar::sub-line{{width:0;height:0;}}")
        self.hscroll.valueChanged.connect(lambda v: self._set_voff(v / 10.0))
        hrl.addWidget(self.hscroll); lo.addWidget(hr)

    def _set_voff(self, v):
        self.voff = v; self._sync()

    def _sync(self):
        self.ruler.bpx = self.bpx; self.ruler.view_off = self.voff; self.ruler.update()
        for ct in self._containers.values():
            ct.lane.set_view(self.bpx, self.voff)
            ct.auto_lane.set_view(self.bpx, self.voff)

    def add_track(self, track: Track):
        ct = TrackContainer(track, self.proj)
        self._wire(ct)
        self.tvb.insertWidget(self.tvb.count() - 2, ct)
        self._containers[track.id] = ct
        ct.lane.set_view(self.bpx, self.voff)
        ct.auto_lane.set_view(self.bpx, self.voff)
        ct.auto_lane.setVisible(self._show_auto)

    def _wire(self, ct: TrackContainer):
        l  = ct.lane; h = ct.header
        l.clip_moved.connect(self._on_moved)
        l.clip_deleted.connect(self._on_deleted)
        l.clip_dropped.connect(self._on_dropped)
        l.clip_resized.connect(self._on_resized)
        l.clip_split.connect(self._on_split)
        l.clip_selected.connect(self._on_selected)
        l.clip_copy_req.connect(self._on_copy)
        l.clip_cut_req.connect(self._on_cut)
        h.mute_changed.connect(lambda tid, v: setattr(self.proj.tracks[tid], "muted", v))
        h.solo_changed.connect(lambda tid, v: setattr(self.proj.tracks[tid], "solo",  v))
        h.volume_changed.connect(lambda tid, v: setattr(self.proj.tracks[tid], "volume", v))
        h.pan_changed.connect(lambda tid, v: setattr(self.proj.tracks[tid], "pan", v))
        h.delete_clicked.connect(self._on_del)
        h.rename_clicked.connect(self._on_rename)
        h.fx_clicked.connect(self._on_fx)
        h.auto_clicked.connect(self._on_auto_toggle)

    def _on_moved(self, cid, ntid, nb):
        c = self.proj.clips.get(cid)
        if not c: return
        if c.track_id != ntid:
            ot = self.proj.tracks.get(c.track_id)
            if ot: ot.clips = [x for x in ot.clips if x != cid]
            c.track_id = ntid
            nt = self.proj.tracks.get(ntid)
            if nt and cid not in nt.clips: nt.clips.append(cid)
        c.start_beat = max(0.0, nb); self.refresh()

    def _on_deleted(self, cid):
        self.proj.remove_clip(cid); self.refresh()

    def _on_dropped(self, sid, tid, b):
        s = self.proj.samples.get(sid)
        if s: self.proj.add_clip(tid, s, b)
        self.refresh()

    def _on_resized(self, cid, l):
        c = self.proj.clips.get(cid)
        if c: c.length_beats = max(0.25, l)

    def _on_split(self, cid, b):
        self.proj.split_clip(cid, b); self.refresh()

    def _on_selected(self, cid):
        self._selected_cid = cid
        for ct in self._containers.values():
            if cid not in ct.track.clips:
                ct.lane.selected_cid = None; ct.lane.update()

    def _on_copy(self, cid):
        src = self.proj.clips.get(cid)
        if not src: return
        self._clipboard = Clip(src.track_id, src.sample, src.start_beat, src.length_beats)
        self._clipboard.gain     = src.gain
        self._clipboard.fade_in  = src.fade_in
        self._clipboard.fade_out = src.fade_out

    def _on_cut(self, cid):
        self._on_copy(cid)
        self.proj.remove_clip(cid)
        self._selected_cid = None
        for ct in self._containers.values():
            ct.lane.selected_cid = None
        self.refresh()

    def paste_clip(self):
        if not self._clipboard: return
        src = self._clipboard; tid = src.track_id
        if tid not in self.proj.tracks: return
        beat = round((src.start_beat + src.length_beats + 0.25) * 4) / 4
        nc = self.proj.add_clip(tid, src.sample, beat, src.length_beats)
        nc.gain = src.gain; nc.fade_in = src.fade_in; nc.fade_out = src.fade_out
        self._selected_cid = nc.id
        ct = self._containers.get(tid)
        if ct: ct.lane.selected_cid = nc.id
        self.refresh()

    def get_selected_clip(self) -> Optional[Clip]:
        return self.proj.clips.get(self._selected_cid)

    def _on_del(self, tid):
        self.proj.remove_track(tid)
        ct = self._containers.pop(tid, None)
        if ct: self.tvb.removeWidget(ct); ct.deleteLater()
        self.track_deleted.emit(tid)

    def _on_rename(self, tid):
        t = self.proj.tracks.get(tid)
        if not t: return
        name, ok = QInputDialog.getText(self, "Rename Track", "Name:", text=t.name)
        if ok and name:
            t.name = name
            ct = self._containers.get(tid)
            if ct: ct.header.name_btn.setText(name[:14])

    def _on_fx(self, tid):
        t = self.proj.tracks.get(tid)
        if t: FXDialog(t, self).exec()

    def _on_auto_toggle(self, tid):
        ct = self._containers.get(tid)
        if ct: ct.auto_lane.setVisible(not ct.auto_lane.isVisible())

    def refresh(self):
        for ct in self._containers.values(): ct.lane.update()

    def set_playhead(self, b):
        self.ruler.set_playhead(b)
        for ct in self._containers.values(): ct.lane.set_playhead(b)

    def set_tool(self, t):
        for ct in self._containers.values(): ct.lane.set_tool(t)

    def set_zoom(self, bpx):
        self.bpx = bpx; self._sync()

    def toggle_auto(self, v):
        self._show_auto = v
        for ct in self._containers.values(): ct.auto_lane.setVisible(v)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(max(8, min(200, self.bpx + e.angleDelta().y() * 0.05)))
            e.accept()
        else:
            super().wheelEvent(e)


# ══════════════════════════════════════════════════════════
#  EXPORT DIALOG
# ══════════════════════════════════════════════════════════
class ExportDialog(QDialog):
    def __init__(self, engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Export Mix — LimeSoda"); self.setFixedSize(360, 215)
        self.setStyleSheet(
            f"QDialog{{background:{PANEL2.name()};color:{TEXT.name()};}}"
            f"QLabel{{color:{TEXT.name()};font-size:12px;}}")
        lo = QVBoxLayout(self); lo.setSpacing(12); lo.setContentsMargins(22, 18, 22, 18)
        t = QLabel("EXPORT MIX"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(f"color:{ACCENT2.name()};font-size:14px;font-weight:700;letter-spacing:3px;")
        lo.addWidget(t)
        fmts = QHBoxLayout(); self.fmt_btns = {}
        for fmt in ["WAV", "OGG", "FLAC"]:
            b = mk_btn(fmt, hover=ACCENT2.name(), chk_bg="#0e1a28", w=90, checkable=True)
            b.setChecked(fmt == "WAV"); fmts.addWidget(b); self.fmt_btns[fmt] = b
            b.clicked.connect(lambda _, f=fmt: self._sel(f))
        lo.addLayout(fmts)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setFixedHeight(8); self.prog.setTextVisible(False)
        self.prog.setStyleSheet(
            f"QProgressBar{{background:{BORDER.name()};border-radius:4px;border:none;}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {ACCENT.name()},stop:1 {ACCENT2.name()});border-radius:4px;}}")
        lo.addWidget(self.prog)
        self.st = QLabel("Choose format and click Export.")
        self.st.setStyleSheet(f"color:{TEXTDIM.name()};font-size:10px;font-family:monospace;")
        lo.addWidget(self.st)
        btns = QHBoxLayout()
        c = mk_btn("Cancel", hover=ACCENT3.name(), w=74); c.clicked.connect(self.reject)
        self.go = mk_btn("EXPORT", hover=ACCENT2.name(), w=90)
        self.go.clicked.connect(self._go); btns.addWidget(c); btns.addStretch(); btns.addWidget(self.go)
        lo.addLayout(btns)

    def _sel(self, fmt):
        for f, b in self.fmt_btns.items(): b.setChecked(f == fmt)

    def _fmt(self):
        for f, b in self.fmt_btns.items():
            if b.isChecked(): return f
        return "WAV"

    def _go(self):
        fmt = self._fmt()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save", f"limesoda.{fmt.lower()}",
            f"{fmt} (*.{fmt.lower()});;All Files (*)")
        if not path: return
        self.go.setEnabled(False); self.st.setText("Rendering…"); self.prog.setValue(5)

        class W(QThread):
            ps = pyqtSignal(int); ds = pyqtSignal(bool, str)
            def __init__(s2, eng, p, f):
                super().__init__(); s2.eng = eng; s2.p = p; s2.f = f
            def run(s2):
                try:
                    mix = s2.eng.render_full(lambda v: s2.ps.emit(v))
                    s2.ps.emit(90)
                    ExportDialog._write(mix, SR, s2.p, s2.f)
                    s2.ps.emit(100); s2.ds.emit(True, "")
                except Exception as ex: s2.ds.emit(False, str(ex))

        self.w = W(self.engine, path, fmt)
        self.w.ps.connect(self.prog.setValue)
        self.w.ds.connect(self._done); self.w.start()

    def _done(self, ok, err):
        if ok: self.st.setText("✓ Export complete!"); QTimer.singleShot(1400, self.accept)
        else: self.st.setText(f"Error: {err}"); self.go.setEnabled(True)

    @staticmethod
    def _write(mix: np.ndarray, sr: int, path: str, fmt: str):
        peak = np.max(np.abs(mix))
        if peak > 0: mix = mix / peak
        fmt = fmt.upper()
        if fmt == "WAV":
            pcm = (mix * 32767).astype("<i2"); raw = pcm.flatten().tobytes()
            nc = 2; bps = 16; ba = nc * bps // 8; br = sr * ba
            with open(path, "wb") as f:
                f.write(b"RIFF"); f.write(struct.pack("<I", 36 + len(raw)))
                f.write(b"WAVE"); f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, nc, sr, br, ba, bps))
                f.write(b"data"); f.write(struct.pack("<I", len(raw))); f.write(raw)
        elif AUDIO_OK:
            fmt_map = {"OGG": ("OGG", "VORBIS"), "FLAC": ("FLAC", "PCM_24")}
            ff, sub = fmt_map.get(fmt, ("WAV", "PCM_16"))
            sf.write(path, mix, sr, format=ff, subtype=sub)
        else:
            raise RuntimeError("soundfile not installed — only WAV export available")


# ══════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════
class LimeSoda(QMainWindow):
    def __init__(self):
        super().__init__()
        self.proj   = Project()
        self.engine = AudioEngine(self.proj)
        self.engine.tick.connect(self._on_tick)
        self._tool  = "select"
        self.setWindowTitle("LimeSoda DAW")
        self.resize(1440, 820); self.setMinimumSize(940, 580)
        self._apply_style()
        self._build_ui()
        self._ph_timer = QTimer(self)
        self._ph_timer.setInterval(16)
        self._ph_timer.timeout.connect(self._update_ph)
        self._ph_timer.start()

    # ── global stylesheet ──────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget {{ background:{BG.name()}; color:{TEXT.name()}; }}
            QSplitter::handle {{ background:{BORDER.name()}; width:1px; height:1px; }}
            QTabWidget::pane  {{ border:1px solid {BORDER.name()}; border-radius:4px; }}
            QTabBar::tab      {{ background:{PANEL2.name()}; color:{TEXTDIM.name()};
                                 border:1px solid {BORDER.name()};
                                 padding:5px 16px; border-radius:3px 3px 0 0;
                                 font-size:11px; font-weight:600; }}
            QTabBar::tab:selected {{ background:{PANEL.name()}; color:{ACCENT2.name()};
                                     border-bottom-color:{PANEL.name()}; }}
            QTabBar::tab:hover    {{ color:{TEXT.name()}; }}
            QStatusBar {{ background:#0a0a10; border-top:1px solid {BORDER.name()};
                          color:{TEXTDIM.name()}; font-size:10px; font-family:monospace; }}
            QMenuBar   {{ background:{PANEL.name()}; color:{TEXT.name()};
                          border-bottom:1px solid {BORDER.name()}; }}
            QMenuBar::item:selected {{ background:#25253a; }}
            QMenu      {{ background:{PANEL2.name()}; border:1px solid {BORDER.name()};
                          color:{TEXT.name()}; }}
            QMenu::item:selected {{ background:#25253a; }}
            QToolBar   {{ background:qlineargradient(y1:0,y2:1,stop:0 #1a1a24,stop:1 {PANEL.name()});
                          border-bottom:1px solid {BORDER.name()};
                          spacing:4px; padding:4px 8px; }}
            QToolBar::separator {{ background:{BORDER.name()}; width:1px; margin:4px 2px; }}
            QSpinBox   {{ background:#0d0d14; border:1px solid {BORDER.name()};
                          color:{ACCENT.name()}; font-family:'Courier New';
                          font-size:13px; font-weight:700; border-radius:4px; padding:2px 4px; }}
            QSpinBox::up-button,QSpinBox::down-button {{ width:14px; background:#1a1a28; border:none; }}
        """)

    # ── UI construction ────────────────────────────────────
    def _build_ui(self):
        self._build_menubar()
        self._build_toolbar()
        central = QWidget(); self.setCentralWidget(central)
        lo = QHBoxLayout(central); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)

        self.library = SampleLibrary(self.proj, self.engine); lo.addWidget(self.library)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color:{BORDER.name()};"); lo.addWidget(sep)

        self.tabs = QTabWidget(); lo.addWidget(self.tabs)

        self.arrange = Arrangement(self.proj)
        self.arrange.track_deleted.connect(self._on_track_del)
        self.arrange.add_track_req.connect(self._add_empty_track)
        self.arrange.seek_signal.connect(self.engine.seek)
        self.tabs.addTab(self.arrange, "🎛  ARRANGEMENT")

        self.beat_ed = BeatEditor(self.proj, self.engine)
        self.tabs.addTab(self.beat_ed, "🥁  BEAT EDITOR")

        self.piano = PianoRoll(self.proj, self.engine)
        self.tabs.addTab(self.piano, "🎹  PIANO ROLL")

        self.mixer = MixerPanel(self.proj, self.engine)
        self.tabs.addTab(self.mixer, "🎚  MIXER")

        self._build_statusbar()

    def _build_menubar(self):
        mb = self.menuBar()

        def act(menu, name, sc, fn):
            a = QAction(name, self); a.setShortcut(sc); a.triggered.connect(fn)
            menu.addAction(a); return a

        fm = mb.addMenu("File")
        act(fm, "Import Audio…",  "Ctrl+O", self._import)
        fm.addSeparator()
        act(fm, "Export Mix…",    "Ctrl+E", self._export)
        fm.addSeparator()
        act(fm, "Quit",           "Ctrl+Q", self.close)

        em = mb.addMenu("Edit")
        act(em, "Undo",           "Ctrl+Z", self._undo)
        act(em, "Redo",           "Ctrl+Y", self._redo)
        em.addSeparator()
        act(em, "Copy Clip",      "Ctrl+C", self._copy)
        act(em, "Cut Clip",       "Ctrl+X", self._cut)
        act(em, "Paste Clip",     "Ctrl+V", self._paste)
        em.addSeparator()
        act(em, "Delete Clip",    "Del",    self._delete)
        em.addSeparator()
        act(em, "Add Empty Track","Ctrl+T", self._add_empty_track)
        act(em, "Rename Track",   "F2",     self._rename_current)

        vm = mb.addMenu("View")
        act(vm, "Zoom In",  "Ctrl+=", lambda: self._zoom(1.25))
        act(vm, "Zoom Out", "Ctrl+-", lambda: self._zoom(0.80))
        vm.addSeparator()
        aa = QAction("Show All Automation Lanes", self, checkable=True)
        aa.triggered.connect(lambda v: self.arrange.toggle_auto(v)); vm.addAction(aa)
        vm.addSeparator()
        act(vm, "Arrangement",  "Ctrl+1", lambda: self.tabs.setCurrentIndex(0))
        act(vm, "Beat Editor",  "Ctrl+2", lambda: self.tabs.setCurrentIndex(1))
        act(vm, "Piano Roll",   "Ctrl+3", lambda: self.tabs.setCurrentIndex(2))
        act(vm, "Mixer",        "Ctrl+4", lambda: self.tabs.setCurrentIndex(3))

        tm = mb.addMenu("Track")
        act(tm, "Add Track",       "Ctrl+T",  self._add_empty_track)
        act(tm, "Open Piano Roll", "Ctrl+P",  self._open_piano_roll)

    def _build_toolbar(self):
        tb = self.addToolBar("Main"); tb.setMovable(False); tb.setFixedHeight(46)

        logo = QLabel()
        logo.setText(
            '<span style="color:#00F5A0;font-family:Courier New;font-size:15px;'
            'font-weight:900;letter-spacing:3px;">LIME</span>'
            '<span style="color:#FF6B35;font-family:Courier New;font-size:15px;'
            'font-weight:900;letter-spacing:3px;">SODA</span>')
        logo.setTextFormat(Qt.TextFormat.RichText)
        tb.addWidget(logo); tb.addSeparator()

        # Transport
        self.btn_stop = mk_btn("■  STOP", fg=ACCENT3.name(), hover=ACCENT3.name(), w=76)
        self.btn_stop.clicked.connect(self._on_stop); tb.addWidget(self.btn_stop)
        self.btn_play = mk_btn("▶  PLAY", fg=ACCENT.name(), hover=ACCENT.name(),
                                w=82, checkable=True)
        self.btn_play.toggled.connect(self._on_play); tb.addWidget(self.btn_play)
        self.btn_loop = mk_btn("↺  LOOP", fg=ACCENT4.name(), hover=ACCENT4.name(),
                                w=78, checkable=True)
        self.btn_loop.toggled.connect(lambda v: setattr(self.engine, "looping", v))
        tb.addWidget(self.btn_loop); tb.addSeparator()

        # BPM + time sig
        def dim(txt):
            l = QLabel(txt)
            l.setStyleSheet(f"color:{TEXTDIM.name()};font-size:10px;font-family:monospace;letter-spacing:1px;")
            return l

        tb.addWidget(dim("BPM"))
        self.bpm_spin = QSpinBox(); self.bpm_spin.setRange(20, 999)
        self.bpm_spin.setValue(120); self.bpm_spin.setFixedWidth(62)
        self.bpm_spin.valueChanged.connect(self._on_bpm); tb.addWidget(self.bpm_spin)
        tb.addWidget(dim("SIG"))
        self.ts_num = QSpinBox(); self.ts_num.setRange(1, 16); self.ts_num.setValue(4)
        self.ts_num.setFixedWidth(38)
        self.ts_num.setStyleSheet(self.bpm_spin.styleSheet().replace(ACCENT.name(), ACCENT2.name()))
        self.ts_num.valueChanged.connect(lambda v: setattr(self.proj, "ts_num", v))
        self.ts_den = QSpinBox(); self.ts_den.setRange(1, 16); self.ts_den.setValue(4)
        self.ts_den.setFixedWidth(38)
        self.ts_den.setStyleSheet(self.bpm_spin.styleSheet().replace(ACCENT.name(), ACCENT2.name()))
        self.ts_den.valueChanged.connect(lambda v: setattr(self.proj, "ts_den", v))
        tb.addWidget(self.ts_num); tb.addWidget(dim("/")); tb.addWidget(self.ts_den)
        tb.addSeparator()

        # Position display
        self.pos_lbl = QLabel("001 : 1 : 00")
        self.pos_lbl.setStyleSheet(
            f"font-family:'Courier New';font-size:12px;color:{ACCENT2.name()};"
            f"background:#0d0d14;border:1px solid {BORDER.name()};"
            f"border-radius:4px;padding:3px 8px;")
        tb.addWidget(self.pos_lbl); tb.addSeparator()

        # Master volume
        tb.addWidget(dim("VOL"))
        self.vol_sl = mk_slider(ACCENT.name(), lo=0, hi=100, val=80, w=72)
        self.vol_lbl = QLabel("80%")
        self.vol_lbl.setStyleSheet(f"color:{ACCENT.name()};font-family:monospace;font-size:10px;min-width:30px;")
        self.vol_sl.valueChanged.connect(lambda v: self.vol_lbl.setText(f"{v}%"))
        tb.addWidget(self.vol_sl); tb.addWidget(self.vol_lbl); tb.addSeparator()

        # Tools
        tb.addWidget(dim("TOOL"))
        self.t_sel = mk_btn("⬡  SELECT", hover=ACCENT2.name(), chk_bg="#0e1a28", w=90, checkable=True)
        self.t_sel.setChecked(True)
        self.t_sel.clicked.connect(lambda: self._set_tool("select")); tb.addWidget(self.t_sel)
        self.t_cut = mk_btn("✂  CUT", hover=ACCENT3.name(), chk_bg="#2e1a0a", w=76, checkable=True)
        self.t_cut.clicked.connect(lambda: self._set_tool("cut")); tb.addWidget(self.t_cut)
        tb.addSeparator()

        # Zoom
        tb.addWidget(dim("ZOOM"))
        self.zm_out = mk_btn("−", w=26, h=26)
        self.zm_out.clicked.connect(lambda: self._zoom(0.80)); tb.addWidget(self.zm_out)
        self.zm_sl = mk_slider(ACCENT2.name(), lo=8, hi=200, val=DEFAULT_BPX, w=78)
        self.zm_sl.valueChanged.connect(self._on_zoom); tb.addWidget(self.zm_sl)
        self.zm_in = mk_btn("+", w=26, h=26)
        self.zm_in.clicked.connect(lambda: self._zoom(1.25)); tb.addWidget(self.zm_in)
        self.zm_lbl = QLabel("1.0×")
        self.zm_lbl.setStyleSheet(f"color:{ACCENT2.name()};font-family:monospace;font-size:10px;min-width:32px;")
        tb.addWidget(self.zm_lbl); tb.addSeparator()

        # Import / Export
        imp = mk_btn("⬇  IMPORT", fg=ACCENT.name(),  hover=ACCENT.name(),  w=100)
        imp.clicked.connect(self._import); tb.addWidget(imp)
        exp = mk_btn("⬆  EXPORT", fg=ACCENT2.name(), hover=ACCENT2.name(), w=100)
        exp.clicked.connect(self._export); tb.addWidget(exp)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.st_tr = QLabel("TRACKS: 0")
        self.st_cl = QLabel("CLIPS: 0")
        self.st_sm = QLabel("SAMPLES: 0")
        self.st_du = QLabel("DURATION: 0:00")
        hint = QLabel(
            "  Space=Play  V=Select  C=Cut  "
            "Ctrl+C/X/V=Copy/Cut/Paste  Ctrl+Z/Y=Undo/Redo  "
            "Del=Delete  Ctrl+P=Piano Roll  Ctrl+1–4=Tabs  Ctrl+Scroll=Zoom")
        hint.setStyleSheet(f"color:{TEXTDIM.name()};")
        for w in [self.st_tr, self.st_cl, self.st_sm, self.st_du]:
            w.setStyleSheet(f"color:{TEXT.name()};padding:0 10px;font-family:monospace;font-size:10px;")
            sb.addWidget(w)
        sb.addPermanentWidget(hint)

    # ── transport slots ────────────────────────────────────
    def _on_play(self, checked: bool):
        if checked:
            self.btn_play.setText("⏸  PAUSE"); self.engine.play()
        else:
            self.btn_play.setText("▶  PLAY");  self.engine.pause()

    def _on_stop(self):
        self.engine.stop()
        self.btn_play.setChecked(False); self.btn_play.setText("▶  PLAY")
        self._on_tick(0.0)

    def _on_tick(self, _): pass

    def _update_ph(self):
        b = self.engine.position; self.arrange.set_playhead(b)
        bar = int(b / 4) + 1; bi = int(b % 4) + 1; tk = int((b % 1) * 96)
        self.pos_lbl.setText(f"{bar:03d} : {bi} : {tk:02d}")

    def _on_bpm(self, v: int):
        self.proj.bpm = v
        if self.engine.playing: self.engine.rebuild()

    # ── file / import ──────────────────────────────────────
    def _import(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Audio", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.aac *.m4a);;All Files (*)")
        for p in paths: self._load(p)

    def _load(self, path: str):
        s = self.proj.add_sample(path)
        if AUDIO_OK and s.data is None:
            del self.proj.samples[s.id]; return
        self.library.add_sample(s)
        t = self.proj.add_track(s.name)
        self.arrange.add_track(t)
        self.proj.add_clip(t.id, s, 0.0)
        self.arrange.refresh()
        self.beat_ed.rebuild()
        self.mixer.rebuild()
        self._stats()
        if self.engine.playing: self.engine.rebuild()

    def _add_empty_track(self):
        t = self.proj.add_track()
        self.arrange.add_track(t)
        self.beat_ed.rebuild()
        self.mixer.rebuild()
        self._stats()

    def _on_track_del(self, _):
        self.beat_ed.rebuild(); self.mixer.rebuild(); self._stats()

    # ── tools ──────────────────────────────────────────────
    def _set_tool(self, tool: str):
        self._tool = tool
        self.t_sel.setChecked(tool == "select")
        self.t_cut.setChecked(tool == "cut")
        self.arrange.set_tool(tool)

    def _zoom(self, f: float):
        self.zm_sl.setValue(max(8, min(200, int(self.arrange.bpx * f))))

    def _on_zoom(self, v: int):
        self.arrange.set_zoom(v); self.zm_lbl.setText(f"{v/60:.1f}×")

    # ── copy / cut / paste / delete ───────────────────────
    def _copy(self):
        c = self.arrange.get_selected_clip()
        if c:
            self.arrange._on_copy(c.id)
            self.statusBar().showMessage(f"Copied: {c.sample.name}", 2000)
        else:
            self.statusBar().showMessage("No clip selected.", 2000)

    def _cut(self):
        c = self.arrange.get_selected_clip()
        if c:
            name = c.sample.name; self.arrange._on_cut(c.id); self._stats()
            if self.engine.playing: self.engine.rebuild()
            self.statusBar().showMessage(f"Cut: {name}", 2000)

    def _paste(self):
        if not self.arrange._clipboard:
            self.statusBar().showMessage("Clipboard is empty.", 2000); return
        self.arrange.paste_clip(); self._stats()
        if self.engine.playing: self.engine.rebuild()
        self.statusBar().showMessage("Pasted clip.", 2000)

    def _delete(self):
        c = self.arrange.get_selected_clip()
        if c:
            self.arrange._on_deleted(c.id); self._stats()
            if self.engine.playing: self.engine.rebuild()

    # ── undo / redo ────────────────────────────────────────
    def _undo(self):
        self.proj.undo(); self.arrange.refresh(); self._stats()
        self.statusBar().showMessage("Undo.", 1500)

    def _redo(self):
        self.proj.redo(); self.arrange.refresh(); self._stats()
        self.statusBar().showMessage("Redo.", 1500)

    # ── piano roll ─────────────────────────────────────────
    def _open_piano_roll(self):
        c = self.arrange.get_selected_clip()
        if c:
            t = self.proj.tracks.get(c.track_id)
            if t:
                self.piano.set_track(t)
                self.tabs.setCurrentWidget(self.piano)
        else:
            self.statusBar().showMessage("Select a clip first (Ctrl+P).", 2000)

    def _rename_current(self):
        c = self.arrange.get_selected_clip()
        if c:
            self.arrange._on_rename(c.track_id)

    # ── export ─────────────────────────────────────────────
    def _export(self):
        if not self.proj.clips:
            QMessageBox.information(self, "Nothing to export", "Add some clips first.")
            return
        ExportDialog(self.engine, self).exec()

    # ── stats bar ──────────────────────────────────────────
    def _stats(self):
        self.st_tr.setText(f"TRACKS: {len(self.proj.tracks)}")
        self.st_cl.setText(f"CLIPS: {len(self.proj.clips)}")
        self.st_sm.setText(f"SAMPLES: {len(self.proj.samples)}")
        sec = self.proj.max_beat() * self.proj.spb() if self.proj.clips else 0
        self.st_du.setText(f"DURATION: {int(sec//60)}:{int(sec%60):02d}")

    # ── keyboard shortcuts ─────────────────────────────────
    def keyPressEvent(self, e):
        k    = e.key()
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if   k == Qt.Key.Key_Space:                          self.btn_play.setChecked(not self.btn_play.isChecked())
        elif k == Qt.Key.Key_V and not ctrl:                 self._set_tool("select")
        elif k == Qt.Key.Key_C and not ctrl:                 self._set_tool("cut")
        elif k == Qt.Key.Key_C and ctrl:                     self._copy()
        elif k == Qt.Key.Key_X and ctrl:                     self._cut()
        elif k == Qt.Key.Key_V and ctrl:                     self._paste()
        elif k == Qt.Key.Key_Z and ctrl:                     self._undo()
        elif k == Qt.Key.Key_Y and ctrl:                     self._redo()
        elif k == Qt.Key.Key_P and ctrl:                     self._open_piano_roll()
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace): self._delete()
        elif k == Qt.Key.Key_F2:                             self._rename_current()
        else: super().keyPressEvent(e)

    # ── window-level drag & drop ───────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a")):
                self._load(p)

    def closeEvent(self, e):
        self.engine.stop(); super().closeEvent(e)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LimeSoda")

    if not AUDIO_OK:
        QMessageBox.warning(
            None, "Missing Libraries",
            "sounddevice / soundfile are not installed.\n"
            "Audio playback and OGG/FLAC export will be disabled.\n\n"
            "Install with:\n    pip install sounddevice soundfile")

    win = LimeSoda()
    win.setAcceptDrops(True)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()