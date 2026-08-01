---
name: Synthesis1
colors:
  surface: '#fcf9f8'
  surface-dim: '#dcd9d9'
  surface-bright: '#fcf9f8'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f2'
  surface-container: '#f0eded'
  surface-container-high: '#eae7e7'
  surface-container-highest: '#e5e2e1'
  on-surface: '#1b1c1c'
  on-surface-variant: '#504442'
  inverse-surface: '#303030'
  inverse-on-surface: '#f3f0ef'
  outline: '#827472'
  outline-variant: '#d3c3c0'
  surface-tint: '#745853'
  primary: '#271310'
  on-primary: '#ffffff'
  primary-container: '#3e2723'
  on-primary-container: '#ae8d87'
  inverse-primary: '#e3beb8'
  secondary: '#8f4e00'
  on-secondary: '#ffffff'
  secondary-container: '#ff8f00'
  on-secondary-container: '#623400'
  tertiary: '#161916'
  on-tertiary: '#ffffff'
  tertiary-container: '#2b2d2b'
  on-tertiary-container: '#939491'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdad4'
  primary-fixed-dim: '#e3beb8'
  on-primary-fixed: '#2b1613'
  on-primary-fixed-variant: '#5b403c'
  secondary-fixed: '#ffdcc2'
  secondary-fixed-dim: '#ffb77a'
  on-secondary-fixed: '#2e1500'
  on-secondary-fixed-variant: '#6d3a00'
  tertiary-fixed: '#e2e3df'
  tertiary-fixed-dim: '#c6c7c3'
  on-tertiary-fixed: '#1a1c1a'
  on-tertiary-fixed-variant: '#454745'
  background: '#fcf9f8'
  on-background: '#1b1c1c'
  surface-variant: '#e5e2e1'
typography:
  display-lg:
    fontFamily: Playfair Display
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Playfair Display
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-lg-mobile:
    fontFamily: Playfair Display
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
  title-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: '1.5'
    letterSpacing: 0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 8px
  container-max: 1440px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
---

## Brand & Style

This design system is built for the high-stakes environments of cybersecurity and medical diagnostics, where precision meets prestige. The aesthetic is rooted in **Modern Editorial Minimalism**—blending the authoritative weight of a luxury heritage brand with the functional clarity required for complex data.

The UI should evoke the "Quiet Luxury" of a high-end hospitality lounge: calm, ordered, and expensive. It utilizes generous whitespace, deliberate typographic hierarchies, and a tactile sense of depth to provide users with a feeling of total control and absolute trust. The interface avoids frantic patterns, favoring steady, rhythmic layouts that allow critical information to breathe.

## Colors

The palette is anchored in a sophisticated "Espresso and Cream" foundation to differentiate from the cold, blue-heavy aesthetics typical of the tech sector.

- **Primary (Deep Coffee):** Used for structural elements like sidebars, headers, and primary actions. It provides a grounded, stable feeling.
- **Secondary (Warm Amber):** Reserved exclusively for high-priority call-to-actions, status alerts, or critical data points. It represents the "glow" of a high-end interior.
- **Surface (Cream/Off-white):** The background is never pure white (#FFFFFF), but rather a warm, non-glare cream to reduce eye strain during long shifts in clinical or security settings.
- **Functional States:** Success should use a desaturated forest green; Error should use a deep carmine red, ensuring they integrate with the warm palette without looking out of place.

## Typography

The typography strategy employs a "High-Contrast Pairing." 

- **Headlines:** Playfair Display is used for page titles and major section headers to establish an editorial, high-end feel. It conveys authority and history.
- **UI & Data:** Inter is used for all functional elements. Its neutral, systematic nature ensures that medical records and security logs remain legible and objective.
- **Labels:** Use `label-caps` for table headers and small metadata categories to create a sense of architectural structure.

## Layout & Spacing

The design system utilizes a **Fixed-Fluid Hybrid Grid**. On desktop, the main content area is capped at 1440px to maintain readability, while the sidebar remains at a fixed width of 280px.

A strict 8px base unit governs all spatial relationships. 
- **Information Density:** For medical dashboards, use "Compact" spacing (8px/12px). For executive overviews, use "Spacious" spacing (24px/32px).
- **Alignment:** All elements must align to the left-edge of the typographic grid. Vertical rhythm is maintained by ensuring line-heights are multiples of 4px.

## Elevation & Depth

This system uses **Tonal Layering** combined with **Ambient Shadows** to create a sense of "physical" objects on a desk.

- **Level 0 (Base):** The Cream background.
- **Level 1 (Cards):** Slightly elevated using a very soft, large-radius shadow (Blur: 20px, Opacity: 4%, Color: Deep Coffee). These cards have a subtle 1px border in a slightly darker cream (#E0E0D5).
- **Level 2 (Popovers/Modals):** Higher elevation with a more pronounced shadow to create focus.
- **Depth Technique:** Use "inner glows" on buttons (1px stroke, top only) to simulate the way light hits the edge of a physical button.

## Shapes

The shape language is **Soft and Architectural**. 

We avoid overly aggressive "pill" shapes which can feel too casual or "app-like." Instead, we use a consistent 4px (Soft) radius for standard UI elements like inputs and buttons. Large cards and containers use 8px (rounded-lg) to feel substantial but precise. The goal is to feel like custom-milled furniture—exact, not soft.

## Components

- **Sleek Sidebar:** The sidebar uses a deep brown (#3E2723) background with "ghost" navigation items. Active states are indicated by a thin Amber (#FF8F00) vertical line on the left and a subtle tonal shift in the background.
- **Polished Cards:** Backgrounds are pure white to pop against the cream page surface. Borders are soft and low-contrast.
- **Refined Buttons:**
    - *Primary:* Solid Deep Brown with white text. On hover, a subtle amber top-border or glow appears.
    - *Secondary:* Cream background with a Deep Brown border and text.
- **Input Fields:** Use a "minimalist" style—only a bottom border that thickens and turns Amber on focus, or a fully enclosed box with a very light cream fill.
- **Data Tables:** High-density, no vertical lines. Horizontal lines should be extremely faint. Use Inter for all numbers to ensure tabular alignment.
- **Iconography:** Use light-weight (2px stroke) monochromatic icons. Never use multi-color icons; let the typography and amber accents carry the visual weight.