---
name: remotion-setup
description: Set up Remotion project for generating nanochat visualization videos
---

# Remotion Setup for nanochat Visualizations

Guide for setting up a Remotion project to create educational videos about LLM training.

## Quick Setup

```bash
# Create new Remotion project
cd ~/projects
npm create video@latest nanochat-visualizations

# When prompted:
# - Template: Blank
# - Package manager: npm

cd nanochat-visualizations
npm install
npm run dev
```

Open http://localhost:3000 to see Remotion Studio.

## Project Structure

After setup, your project should look like:

```
nanochat-visualizations/
├── src/
│   ├── index.ts              # Entry point
│   ├── Root.tsx              # Composition registry
│   └── (your components)     # Video components
├── out/                      # Rendered videos
├── package.json
├── tsconfig.json
└── remotion.config.ts
```

## Install Claude Code Skills

Add Remotion best practices skills:

```bash
npx skills add remotion-dev/skills
```

This installs official Remotion skills that guide Claude Code to generate correct animation code.

## Required Packages

The default template includes everything needed:

```json
{
  "dependencies": {
    "@remotion/cli": "^4.0.0",
    "@remotion/bundler": "^4.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "remotion": "^4.0.0"
  }
}
```

## Create Your First Visualization

### Step 1: Create Component

Create `src/TransformerDiagram.tsx`:

```tsx
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const TransformerDiagram: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fadeIn = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  const scale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  return (
    <AbsoluteFill
      style={{
        background: "#0f0f23",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          opacity: fadeIn,
          transform: `scale(${scale})`,
          color: "#fff",
          fontSize: 60,
          fontWeight: "bold",
        }}
      >
        nanochat Transformer
      </div>
    </AbsoluteFill>
  );
};
```

### Step 2: Register Composition

Update `src/Root.tsx`:

```tsx
import { Composition } from "remotion";
import { TransformerDiagram } from "./TransformerDiagram";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="TransformerDiagram"
        component={TransformerDiagram}
        durationInFrames={150}  // 5 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
```

### Step 3: Preview

```bash
npm run dev
```

Navigate to http://localhost:3000 and select "TransformerDiagram" from the sidebar.

### Step 4: Render

```bash
# MP4 video
npx remotion render TransformerDiagram out/transformer.mp4

# GIF for documentation
npx remotion render TransformerDiagram out/transformer.gif --codec gif

# High quality ProRes
npx remotion render TransformerDiagram out/transformer.mov --codec prores
```

## Remotion Fundamentals

### The Frame-Based Model

Everything in Remotion is a function of the current frame:

```tsx
const frame = useCurrentFrame();  // 0, 1, 2, 3, ... N

// All properties derive from frame number
const opacity = interpolate(frame, [0, 30], [0, 1]);
const position = interpolate(frame, [0, 60], [-100, 0]);
const scale = spring({ frame, fps });
```

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `useCurrentFrame()` | Get current frame | `const frame = useCurrentFrame()` |
| `useVideoConfig()` | Get fps, dimensions | `const { fps, width } = useVideoConfig()` |
| `interpolate()` | Linear mapping | `interpolate(frame, [0, 30], [0, 1])` |
| `spring()` | Physics animation | `spring({ frame, fps })` |

### Always Clamp!

```tsx
// BAD - values can go beyond 0-1
const opacity = interpolate(frame, [0, 30], [0, 1]);

// GOOD - values stay within range
const opacity = interpolate(frame, [0, 30], [0, 1], {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
});
```

### Sequences for Timing

```tsx
import { Sequence } from "remotion";

// Element appears at frame 30
<Sequence from={30}>
  <MyElement />
</Sequence>

// Element appears at frame 30, stays for 60 frames
<Sequence from={30} durationInFrames={60}>
  <MyElement />
</Sequence>
```

## nanochat-Specific Components

### Architecture Box

```tsx
const ArchitectureBox: React.FC<{
  title: string;
  subtitle?: string;
  color: string;
  delay: number;
}> = ({ title, subtitle, color, delay }) => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame - delay, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        background: "#1a1a3a",
        border: `2px solid ${color}`,
        borderRadius: 12,
        padding: 20,
        opacity,
        transform: `translateY(${(1 - opacity) * 20}px)`,
      }}
    >
      <div style={{ color, fontSize: 20, fontWeight: "bold" }}>
        {title}
      </div>
      {subtitle && (
        <div style={{ color: "#888", fontSize: 14, marginTop: 5 }}>
          {subtitle}
        </div>
      )}
    </div>
  );
};
```

### Animated Number

```tsx
const AnimatedNumber: React.FC<{
  value: number;
  delay: number;
  suffix?: string;
}> = ({ value, delay, suffix = "" }) => {
  const frame = useCurrentFrame();

  const animatedValue = interpolate(
    frame - delay,
    [0, 60],
    [0, value],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <span style={{ fontVariantNumeric: "tabular-nums" }}>
      {Math.round(animatedValue).toLocaleString()}{suffix}
    </span>
  );
};

// Usage: <AnimatedNumber value={425000000} suffix=" params" delay={30} />
```

## Output Formats

| Format | Command | Use Case |
|--------|---------|----------|
| MP4 | `--codec h264` (default) | Social media, web |
| GIF | `--codec gif` | Documentation, GitHub |
| WebM | `--codec vp8` | Web with alpha |
| ProRes | `--codec prores` | Editing, archival |

## Tips

1. **Start simple** - Get basic animation working first
2. **Use the timeline** - Scrub through to check all frames
3. **Hot reload** - Changes update instantly in preview
4. **Test renders** - Render short clips to verify output
5. **Reference frame numbers** - Calculate: seconds × fps = frames

## Resources

- [Remotion Documentation](https://www.remotion.dev/docs)
- [Remotion Examples](https://www.remotion.dev/examples)
- [claude-code-remotion](https://github.com/az9713/claude-code-remotion) - Reference project
