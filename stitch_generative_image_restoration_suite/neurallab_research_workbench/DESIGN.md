---
name: NeuralLab Research Workbench
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353942'
  surface-container-lowest: '#0a0e16'
  surface-container-low: '#181c24'
  surface-container: '#1c2028'
  surface-container-high: '#262a33'
  surface-container-highest: '#31353e'
  on-surface: '#dfe2ee'
  on-surface-variant: '#bac9cc'
  inverse-surface: '#dfe2ee'
  inverse-on-surface: '#2c3039'
  outline: '#849396'
  outline-variant: '#3b494c'
  surface-tint: '#00daf3'
  primary: '#c3f5ff'
  on-primary: '#00363d'
  primary-container: '#00e5ff'
  on-primary-container: '#00626e'
  inverse-primary: '#006875'
  secondary: '#d0bcff'
  on-secondary: '#3c0091'
  secondary-container: '#571bc1'
  on-secondary-container: '#c4abff'
  tertiary: '#a8ffd2'
  on-tertiary: '#003824'
  tertiary-container: '#5be9ad'
  on-tertiary-container: '#006645'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#9cf0ff'
  primary-fixed-dim: '#00daf3'
  on-primary-fixed: '#001f24'
  on-primary-fixed-variant: '#004f58'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d0bcff'
  on-secondary-fixed: '#23005c'
  on-secondary-fixed-variant: '#5516be'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0f131c'
  on-background: '#dfe2ee'
  surface-variant: '#31353e'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: 0em
  body-lg:
    fontFamily: Geist
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.03em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 12px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-dense: 0.5rem
  margin: 1.5rem
  margin-mobile: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system targets machine learning researchers, computer vision scientists, and generative AI engineers inspecting complex diffusion pipelines, multimodal restorations, and paired synthesis runs (e.g., Oxford-IIIT Pet Restoration and FS2K Face-to-Sketch). 

The emotional tone balances high-velocity technical rigor with surgical clarity:
- **Scientific Instrument Precision:** Dense data environments must remain legibly stratified, eschewing unnecessary decorative clutter for high information density and zero visual latency.
- **Instrument Dark Mode:** Designed for long-duration laboratory and workstation sessions under controlled lighting environments.
- **Controlled Luminescence:** Accents appear as active data vectors, status telemetry, and tensor maps rather than flat decor.

The design movement combines **Modern High-Tech Minimalist Architecture** with **Precision Instrument Functionalism**: deep slate backplanes, structural hairline grid dividers, subdued tonal surfaces, and crisp functional color coding that highlights weights, routing distributions, confidence scores, and error maps without overwhelming the researcher.

## Colors

The palette is engineered around an ultra-deep, cold neutral foundation that prevents eye fatigue while providing high dynamic range for multi-spectral image inspection, latent space visualizers, and tensor metrics.

### Color Roles & Semantics
- **Primary (`#00E5FF` — Electric Cyan):** Active inference vectors, focus states, interactive crosshairs, selected routing pathways, and real-time generation progress indicators.
- **Secondary (`#8B5CF6` — Spectral Violet):** Latent space embeddings, conditioning prompts, synthesis generation controls, and FS2K feature-alignment highlights.
- **Tertiary (`#10B981` — Precision Emerald):** Checkpoint validation pass states, high confidence bounds (>95%), nominal tensor constraints, and restoration fidelity gains (PSNR/SSIM improvements).
- **Warning / Telemetry (`#F59E0B` — Amber Flare):** VRAM allocation thresholds (>85%), gradient clip warnings, dynamic routing drops, and inference thermal limits.
- **Error / Artifact (`#EF4444` — Crimson Error):** NaN gradient detection, tensor mismatch, OOM exceptions, and SSIM disparity/loss hot-spots.
- **Neutral Core (`#0B0F17` — Deep Slate Void):** Layered background substrates ranging from base canvas (`#070A0F`) to elevated inspection docks (`#111827`) and tertiary toolbars (`#1E293B`). Hairline boundary tokens use `#334155` at low opacity.

## Typography

The typography system relies on a tripartite structural pairing:
1. **Space Grotesk (Display & Structural Hierarchy):** Technical, angular neo-grotesque form designed to structure panel headers, modal titles, and workbench views with futuristic clarity.
2. **Geist (Neutral Ergonomic Body):** Unobtrusive geometric sans optimized for continuous reading of configuration parameters, prompt descriptions, and run metadata.
3. **JetBrains Mono (Metric & Tensor Telemetry):** Strict tabular alignment for tensor dimensions (e.g., `[B, 3, 512, 512]`), execution latencies (`42.4ms`), floating-point values (`0.9842`), and layer parameters.

All numeric metric readouts must enforce tabular figure alignment (`font-variant-numeric: tabular-nums`) to prevent layout shifts during live inference streaming.

## Layout & Spacing

This design system uses a flexible, modular dock-and-viewport layout model. The viewport is treated as an integrated scientific console rather than a conventional scrolling document page.

### Desktop & Multi-Monitor Workbenches (>= 1440px)
- **Three-Tier Workspace Layout:** Persistent collapsible left utility rail (model zoo, datasets, pipeline graphs; default 280px), central dynamic viewports (dual split-screen comparison canvas for degraded vs. restored inputs), and right metric/telemetry dock (parameter adjustment, routing matrices, tensor histograms; default 360px).
- **Gutter Rhythm:** Standard `1rem` between multi-pane panels; `0.5rem` (`gutter-dense`) in micro-parameter grids and filter selector blocks.
- **Canvas Edge Margins:** `1.5rem` from workstation bounds.

### Responsive Reflow & Tablet/Mobile Adaptations
- **Tablet (768px – 1439px):** Inspector dock converts into a slide-over drawer; comparison viewport transitions to an overlaid before/after slider rather than side-by-side split screens.
- **Mobile (< 768px):** Single-column stack with bottom floating switchers between visual results, metric logs, and execution controls. Margins compress to `0.75rem`.

## Elevation & Depth

Visual hierarchy uses **tonal elevation tiers** paired with **subtle micro-borders**, rejecting heavy drop shadows that produce visual haze on dark displays.

### Elevation Hierarchy
- **Level 0 (Backplane Canvas):** `#070A0F` — Pure grounding layer for full workbench canvases.
- **Level 1 (Docked Containers & Split Panes):** `#0F141E` — Outlined by `1px solid rgba(255, 255, 255, 0.06)`. Provides the main perimeter for inspection viewports.
- **Level 2 (Nested Toolbars, Metric Cards, Node Blocks):** `#161F2E` — Hover elevation shifts background to `#1E2B3E` with an internal hairline border of `1px solid rgba(0, 229, 255, 0.15)`.
- **Level 3 (Modals, Context Menus, Tooltips):** `#1A2436` — Enhanced with a crisp dark ambient shadow: `0 8px 32px rgba(0, 0, 0, 0.65)`, complemented by a subtle border: `1px solid rgba(255, 255, 255, 0.12)`.

### Optical Highlights & Data Glow
Glow effects are reserved strictly for active operations:
- **Active Node Routing:** Primary accent stroke receives an inner glow `0 0 12px rgba(0, 229, 255, 0.25)`.
- **Inference In-Progress:** Subtle cyclical breathing boundary animation utilizing tertiary/primary gradients.

## Shapes

The shape architecture relies on an industrial, calibrated geometric rhythm (`roundedness: 1` — Soft, baseline 4px / `0.25rem`):

- **Micro Components (Badges, Monospace Tags, Segmented Switches):** `4px` (`0.25rem`). Ensures dense numeric layouts feel carved and architectural.
- **Containers & Panels (Inspection Windows, Viewport Panes, Model Cards):** `8px` (`0.5rem`). Keeps split boundaries tight and aligned to grid tracks.
- **Modals & Flyout Sheets:** `12px` (`0.75rem`). Distinguishes detached overlays from static dock panels.
- **Interactive Pill Toggles (Status Pins, Step Iterators):** High-density pills use `9999px` strictly when indicating binary toggle modes or real-time online/offline inference state.

## Components

### 1. Buttons & Execution Triggers
- **Primary (Generate / Execute Pipeline):** Solid background of `#00E5FF`, font `#070A0F` (Space Grotesk Bold, 13px), flat surface, hover shifts to subtle brightness lift with `0 0 16px rgba(0, 229, 255, 0.4)`.
- **Secondary (Step Iteration / Abort):** Ghost outline with dark background `#161F2E`, `1px` border of `rgba(255, 255, 255, 0.12)`, text `#F1F5F9`.
- **Icon Actions (Viewport Zoom, Reset Pan, Export Mask):** 32x32px square buttons, roundedness `4px`, hover background `rgba(255, 255, 255, 0.05)`.

### 2. Comparison Viewports & Split Inspection Panes
- **Interactive Dual Canvas:** Side-by-side or curtain wipe between degraded Oxford-IIIT input vs. restored super-resolution output, or FS2K portrait vs. synthesized vector sketch.
- **Center Divider:** 2px solid cyan/white handle with interactive horizontal drag markers, displaying live coordinates `(X, Y)` and zoom scale (`100%`, `400%`, `1600% nearest-neighbor`).
- **Synchronized Pan/Zoom:** Locks multi-spectral tensor views in identical viewing coordinates.

### 3. Metric Badges & Monospace Confidence Tags
- **Confidence Badges:** Pill-shaped background (`rgba(16, 185, 129, 0.1)` for high SSIM/confidence, `rgba(239, 68, 68, 0.1)` for artifact divergence). Font: JetBrains Mono 11px. Prefixed with a 6px status LED.
- **Tensor Shape Indicator:** Dark slate background `#111827`, border `1px solid rgba(255, 255, 255, 0.08)`, text `#94A3B8`. Example: `[1, 512, 512, 3] fp16`.

### 4. Input Fields & Prompt Engineering Consoles
- **Prompt & Negative Prompt Areas:** Surface `#0C111C`, border `1px solid rgba(255, 255, 255, 0.08)`, focus border `1px solid #00E5FF`. Text rendered in Geist 13px with syntax highlighting for token weights (e.g., `(pet:1.2)`, `[sketch_style:0.8]`).
- **Numerical Inputs (CFG Scale, Seed, Steps):** Compact input with step spinners, displaying current step count and latency overhead inline.

### 5. Routing Flow Diagrams & Model Architecture Cards
- **Routing Node Cards:** Dense panels detailing model sub-networks (e.g., CLIP Text Encoder, UNet/DiT Denoising Backbone, VAE Decoder, ControlNet AdaIN).
- **Dynamic Routing Links:** Connecting SVG paths displaying capacity and weight allocation using color intensities: cyan (>70% weight), violet (residual connection), or dashed neutral (bypassed route).

### 6. Sleek Tab Navigation & Layout Switchers
- **Segmented Workbench Switchers:** Clean baseline bar containing low-profile rectangular tabs (`Restoration Lab`, `FS2K Sketch Studio`, `Latency Profiler`, `Checkpoint Diff`).
- **Active State:** Underlined with a crisp 2px bar in `#00E5FF`, font colored `#FFFFFF`, while inactive tabs remain muted at `#64748B`.