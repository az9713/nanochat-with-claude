---
name: visualize
description: Generate animated videos to visualize nanochat architecture, training metrics, and concepts using Remotion
---

# nanochat Visualization Skill

Generate educational videos to visualize LLM training concepts using Remotion.

## Overview

This skill enables creation of animated videos for:
- **Architecture diagrams** - Transformer, attention, MLP animations
- **Training metrics** - Loss curves, MFU, throughput charts
- **Scaling laws** - Model size vs compute visualizations
- **Data flow** - Token processing, embedding, generation

## Prerequisites

Ensure Remotion is set up:

```bash
# Create visualization project (one-time)
cd ~/nanochat-visualizations
npm create video@latest -- --template blank

# Or use existing project
npm install
npm run dev  # Preview at localhost:3000
```

## Available Visualizations

### 1. Transformer Architecture Animation

Animates the full GPT architecture with data flowing through layers.

```tsx
// src/TransformerArchitecture.tsx
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Sequence,
} from "remotion";

export const TransformerArchitecture: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animation phases
  const inputPhase = interpolate(frame, [0, 30], [0, 1], { extrapolateRight: "clamp" });
  const embeddingPhase = interpolate(frame, [30, 60], [0, 1], { extrapolateRight: "clamp" });
  const attentionPhase = interpolate(frame, [60, 120], [0, 1], { extrapolateRight: "clamp" });
  const mlpPhase = interpolate(frame, [120, 180], [0, 1], { extrapolateRight: "clamp" });
  const outputPhase = interpolate(frame, [180, 210], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{
      background: "linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%)",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Title */}
      <div style={{
        position: "absolute",
        top: 40,
        left: 60,
        fontSize: 48,
        fontWeight: "bold",
        color: "#ffffff",
        opacity: inputPhase,
      }}>
        GPT Transformer Architecture
      </div>

      {/* Input tokens */}
      <Sequence from={0} durationInFrames={240}>
        <div style={{
          position: "absolute",
          left: 200,
          top: 200,
          opacity: inputPhase,
          transform: `translateY(${(1 - inputPhase) * 50}px)`,
        }}>
          <div style={{ color: "#da7756", fontSize: 24, marginBottom: 10 }}>
            Input Tokens
          </div>
          <div style={{
            display: "flex",
            gap: 10,
          }}>
            {["Why", "is", "the", "sky", "blue", "?"].map((token, i) => (
              <div key={i} style={{
                background: "#2a2a4a",
                padding: "10px 15px",
                borderRadius: 8,
                color: "#fff",
                fontSize: 18,
                opacity: interpolate(frame - i * 3, [0, 15], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
              }}>
                {token}
              </div>
            ))}
          </div>
        </div>
      </Sequence>

      {/* Embedding layer */}
      <Sequence from={30} durationInFrames={210}>
        <div style={{
          position: "absolute",
          left: 200,
          top: 350,
          opacity: embeddingPhase,
        }}>
          <div style={{
            background: "#3a3a6a",
            padding: 20,
            borderRadius: 12,
            width: 600,
            textAlign: "center",
          }}>
            <div style={{ color: "#f4a261", fontSize: 20 }}>
              Token + Position Embedding
            </div>
            <div style={{ color: "#888", fontSize: 14, marginTop: 8 }}>
              vocab_size × hidden_dim (50257 × {24 * 64})
            </div>
          </div>
        </div>
      </Sequence>

      {/* Transformer blocks */}
      <Sequence from={60} durationInFrames={180}>
        <div style={{
          position: "absolute",
          left: 200,
          top: 450,
          opacity: attentionPhase,
        }}>
          {[0, 1, 2].map((layer) => (
            <div key={layer} style={{
              background: "#2a2a5a",
              padding: 15,
              borderRadius: 10,
              marginBottom: 10,
              width: 600,
              opacity: interpolate(frame - 60 - layer * 15, [0, 15], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#da7756" }}>Layer {layer + 1}</span>
                <span style={{ color: "#888", fontSize: 14 }}>
                  Self-Attention → MLP
                </span>
              </div>
            </div>
          ))}
          <div style={{ color: "#666", textAlign: "center", marginTop: 10 }}>
            ... (24 layers total)
          </div>
        </div>
      </Sequence>

      {/* Output */}
      <Sequence from={180} durationInFrames={60}>
        <div style={{
          position: "absolute",
          left: 200,
          top: 700,
          opacity: outputPhase,
        }}>
          <div style={{
            background: "#4a4a7a",
            padding: 20,
            borderRadius: 12,
            width: 600,
            textAlign: "center",
          }}>
            <div style={{ color: "#4ade80", fontSize: 20 }}>
              Output Logits → Next Token
            </div>
          </div>
        </div>
      </Sequence>

      {/* nanochat branding */}
      <div style={{
        position: "absolute",
        bottom: 30,
        right: 40,
        color: "#666",
        fontSize: 16,
      }}>
        nanochat • depth=24
      </div>
    </AbsoluteFill>
  );
};
```

### 2. Training Loss Animation

Animates a training loss curve with annotations.

```tsx
// src/TrainingLoss.tsx
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";

const lossData = [
  { step: 0, loss: 10.5 },
  { step: 1000, loss: 4.2 },
  { step: 2000, loss: 3.5 },
  { step: 5000, loss: 3.1 },
  { step: 10000, loss: 2.9 },
  { step: 20000, loss: 2.7 },
  { step: 30000, loss: 2.65 },
];

export const TrainingLoss: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const progress = interpolate(frame, [30, 300], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  const chartWidth = 1400;
  const chartHeight = 600;
  const chartLeft = (width - chartWidth) / 2;
  const chartTop = 200;

  // Scale functions
  const xScale = (step: number) => (step / 30000) * chartWidth + chartLeft;
  const yScale = (loss: number) => chartTop + chartHeight - ((loss - 2) / 9) * chartHeight;

  // Generate path
  const visiblePoints = lossData.filter((_, i) => i / lossData.length <= progress);
  const pathD = visiblePoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.step)} ${yScale(p.loss)}`)
    .join(" ");

  return (
    <AbsoluteFill style={{
      background: "#0f0f23",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Title */}
      <div style={{
        position: "absolute",
        top: 50,
        left: chartLeft,
        fontSize: 42,
        fontWeight: "bold",
        color: "#fff",
      }}>
        Training Loss Curve
      </div>
      <div style={{
        position: "absolute",
        top: 100,
        left: chartLeft,
        fontSize: 20,
        color: "#888",
      }}>
        nanochat depth=24 on FineWeb-Edu
      </div>

      {/* Y-axis */}
      <div style={{
        position: "absolute",
        left: chartLeft - 60,
        top: chartTop,
        height: chartHeight,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
      }}>
        {[10, 8, 6, 4, 2].map((v) => (
          <div key={v} style={{ color: "#666", fontSize: 14 }}>{v}</div>
        ))}
      </div>

      {/* X-axis */}
      <div style={{
        position: "absolute",
        left: chartLeft,
        top: chartTop + chartHeight + 20,
        width: chartWidth,
        display: "flex",
        justifyContent: "space-between",
      }}>
        {[0, 10, 20, 30].map((v) => (
          <div key={v} style={{ color: "#666", fontSize: 14 }}>{v}K</div>
        ))}
      </div>

      {/* Chart area */}
      <svg
        width={width}
        height={height}
        style={{ position: "absolute", top: 0, left: 0 }}
      >
        {/* Grid lines */}
        {[2, 4, 6, 8, 10].map((loss) => (
          <line
            key={loss}
            x1={chartLeft}
            x2={chartLeft + chartWidth}
            y1={yScale(loss)}
            y2={yScale(loss)}
            stroke="#333"
            strokeWidth={1}
          />
        ))}

        {/* Loss curve */}
        <path
          d={pathD}
          fill="none"
          stroke="#da7756"
          strokeWidth={4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Current point */}
        {visiblePoints.length > 0 && (
          <circle
            cx={xScale(visiblePoints[visiblePoints.length - 1].step)}
            cy={yScale(visiblePoints[visiblePoints.length - 1].loss)}
            r={8}
            fill="#da7756"
          />
        )}
      </svg>

      {/* Current loss display */}
      <div style={{
        position: "absolute",
        top: chartTop + 20,
        right: chartLeft,
        background: "#2a2a4a",
        padding: "15px 25px",
        borderRadius: 10,
      }}>
        <div style={{ color: "#888", fontSize: 14 }}>Current Loss</div>
        <div style={{ color: "#da7756", fontSize: 36, fontWeight: "bold" }}>
          {(interpolate(progress, [0, 1], [10.5, 2.65])).toFixed(2)}
        </div>
      </div>

      {/* Labels */}
      <div style={{
        position: "absolute",
        bottom: 80,
        left: chartLeft,
        width: chartWidth,
        textAlign: "center",
        color: "#666",
        fontSize: 16,
      }}>
        Training Steps
      </div>
    </AbsoluteFill>
  );
};
```

### 3. Attention Visualization

Animates attention patterns between tokens.

```tsx
// src/AttentionVisualization.tsx
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const tokens = ["The", "sky", "is", "blue", "because"];

// Simulated attention weights (query token "blue" attending to others)
const attentionWeights = [0.05, 0.35, 0.15, 0.40, 0.05];

export const AttentionVisualization: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{
      background: "#0f0f23",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Title */}
      <div style={{
        position: "absolute",
        top: 60,
        left: "50%",
        transform: "translateX(-50%)",
        fontSize: 42,
        fontWeight: "bold",
        color: "#fff",
        textAlign: "center",
      }}>
        Self-Attention Mechanism
      </div>
      <div style={{
        position: "absolute",
        top: 120,
        left: "50%",
        transform: "translateX(-50%)",
        fontSize: 20,
        color: "#888",
      }}>
        How "blue" attends to previous tokens
      </div>

      {/* Token row */}
      <div style={{
        position: "absolute",
        top: 300,
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        gap: 60,
      }}>
        {tokens.map((token, i) => {
          const isQuery = i === 3; // "blue" is the query
          const weight = attentionWeights[i];
          const animProgress = interpolate(frame - 60 - i * 10, [0, 30], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          return (
            <div key={i} style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 20,
            }}>
              {/* Attention bar */}
              <div style={{
                width: 60,
                height: 200,
                background: "#1a1a3a",
                borderRadius: 8,
                position: "relative",
                overflow: "hidden",
              }}>
                <div style={{
                  position: "absolute",
                  bottom: 0,
                  width: "100%",
                  height: `${weight * 100 * animProgress}%`,
                  background: isQuery
                    ? "linear-gradient(180deg, #4ade80, #22c55e)"
                    : `rgba(218, 119, 86, ${weight + 0.3})`,
                  borderRadius: 8,
                }} />
              </div>

              {/* Token */}
              <div style={{
                background: isQuery ? "#22c55e" : "#2a2a5a",
                padding: "12px 20px",
                borderRadius: 10,
                color: "#fff",
                fontSize: 20,
                fontWeight: isQuery ? "bold" : "normal",
              }}>
                {token}
              </div>

              {/* Weight label */}
              <div style={{
                color: "#888",
                fontSize: 16,
                opacity: animProgress,
              }}>
                {(weight * animProgress).toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Explanation */}
      <div style={{
        position: "absolute",
        bottom: 100,
        left: "50%",
        transform: "translateX(-50%)",
        background: "#1a1a3a",
        padding: "20px 40px",
        borderRadius: 12,
        maxWidth: 800,
        textAlign: "center",
      }}>
        <div style={{ color: "#f4a261", fontSize: 18 }}>
          Query: "blue" → Attends most strongly to "sky" (0.35) and itself (0.40)
        </div>
        <div style={{ color: "#666", fontSize: 14, marginTop: 10 }}>
          Attention weights sum to 1.0 via softmax
        </div>
      </div>
    </AbsoluteFill>
  );
};
```

### 4. Scaling Laws Visualization

Animates the relationship between model size and performance.

```tsx
// src/ScalingLaws.tsx
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  Easing,
} from "remotion";

const models = [
  { depth: 12, params: "106M", loss: 3.2, x: 200, y: 500 },
  { depth: 16, params: "188M", loss: 3.0, x: 400, y: 430 },
  { depth: 20, params: "295M", loss: 2.85, x: 600, y: 380 },
  { depth: 24, params: "425M", loss: 2.65, x: 800, y: 320 },
  { depth: 32, params: "756M", loss: 2.45, x: 1000, y: 260 },
  { depth: 48, params: "1.7B", loss: 2.25, x: 1200, y: 200 },
];

export const ScalingLaws: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{
      background: "#0f0f23",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Title */}
      <div style={{
        position: "absolute",
        top: 50,
        left: 100,
        fontSize: 42,
        fontWeight: "bold",
        color: "#fff",
      }}>
        nanochat Scaling Laws
      </div>
      <div style={{
        position: "absolute",
        top: 110,
        left: 100,
        fontSize: 20,
        color: "#888",
      }}>
        Bigger models → Lower loss (power law relationship)
      </div>

      {/* Axes labels */}
      <div style={{
        position: "absolute",
        left: 100,
        top: 350,
        color: "#666",
        fontSize: 16,
        transform: "rotate(-90deg)",
        transformOrigin: "left center",
      }}>
        Training Loss ↓
      </div>
      <div style={{
        position: "absolute",
        bottom: 50,
        left: "50%",
        transform: "translateX(-50%)",
        color: "#666",
        fontSize: 16,
      }}>
        Model Size (Parameters) →
      </div>

      {/* Data points */}
      {models.map((model, i) => {
        const animDelay = i * 15;
        const pointProgress = interpolate(frame - 60 - animDelay, [0, 30], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.out(Easing.back(1.5)),
        });

        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: model.x,
              top: model.y,
              transform: `scale(${pointProgress})`,
              opacity: pointProgress,
            }}
          >
            {/* Point */}
            <div style={{
              width: 24,
              height: 24,
              borderRadius: "50%",
              background: "#da7756",
              boxShadow: "0 0 20px rgba(218, 119, 86, 0.5)",
            }} />

            {/* Label */}
            <div style={{
              position: "absolute",
              top: 35,
              left: "50%",
              transform: "translateX(-50%)",
              textAlign: "center",
              whiteSpace: "nowrap",
            }}>
              <div style={{ color: "#fff", fontSize: 16, fontWeight: "bold" }}>
                d={model.depth}
              </div>
              <div style={{ color: "#888", fontSize: 12 }}>
                {model.params}
              </div>
              <div style={{ color: "#4ade80", fontSize: 14 }}>
                loss={model.loss}
              </div>
            </div>
          </div>
        );
      })}

      {/* Trend line */}
      <svg
        width={1920}
        height={1080}
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      >
        <path
          d={`M ${models.map(m => `${m.x + 12} ${m.y + 12}`).join(" L ")}`}
          fill="none"
          stroke="#da7756"
          strokeWidth={2}
          strokeDasharray="8 4"
          opacity={interpolate(frame, [150, 180], [0, 0.5], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          })}
        />
      </svg>

      {/* Key insight */}
      <div style={{
        position: "absolute",
        right: 100,
        top: 200,
        background: "#1a1a3a",
        padding: 25,
        borderRadius: 12,
        maxWidth: 400,
        opacity: interpolate(frame, [200, 230], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
      }}>
        <div style={{ color: "#f4a261", fontSize: 18, marginBottom: 10 }}>
          Key Insight
        </div>
        <div style={{ color: "#fff", fontSize: 16 }}>
          Each doubling of parameters reduces loss by ~0.15-0.2
        </div>
        <div style={{ color: "#888", fontSize: 14, marginTop: 10 }}>
          GPT-2 level (CORE &gt; 0.256) achieved at d=24
        </div>
      </div>
    </AbsoluteFill>
  );
};
```

## Root.tsx Registration

Register all compositions:

```tsx
// src/Root.tsx
import { Composition, Folder } from "remotion";
import { TransformerArchitecture } from "./TransformerArchitecture";
import { TrainingLoss } from "./TrainingLoss";
import { AttentionVisualization } from "./AttentionVisualization";
import { ScalingLaws } from "./ScalingLaws";

export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="nanochat">
      <Composition
        id="TransformerArchitecture"
        component={TransformerArchitecture}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="TrainingLoss"
        component={TrainingLoss}
        durationInFrames={360}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="AttentionVisualization"
        component={AttentionVisualization}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="ScalingLaws"
        component={ScalingLaws}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
      />
    </Folder>
  );
};
```

## Render Commands

```bash
# Preview
npm run dev

# Render individual videos
npx remotion render TransformerArchitecture out/transformer.mp4
npx remotion render TrainingLoss out/training-loss.mp4
npx remotion render AttentionVisualization out/attention.mp4
npx remotion render ScalingLaws out/scaling-laws.mp4

# Render as GIF for documentation
npx remotion render TransformerArchitecture out/transformer.gif --codec gif
```

## Color Palette (nanochat theme)

| Color | Hex | Usage |
|-------|-----|-------|
| Background Dark | `#0f0f23` | Video background |
| Background Mid | `#1a1a3a` | Cards, panels |
| Claude Orange | `#da7756` | Primary accent, highlights |
| Warm Orange | `#f4a261` | Secondary accent |
| Success Green | `#4ade80` | Positive values |
| Text White | `#ffffff` | Primary text |
| Text Gray | `#888888` | Secondary text |

## Animation Best Practices

1. **Use interpolate with clamp** - Prevents values from overshooting
2. **Stagger animations** - Add delay for sequential elements
3. **Use spring for entrances** - Natural, bouncy feel
4. **Use Easing.out for exits** - Smooth deceleration
5. **Keep videos short** - 10-30 seconds for attention

## Integration with nanochat

When explaining architecture, offer to generate visualizations:

```
User: Explain how attention works in nanochat

Claude: [Explains attention mechanism]

Would you like me to generate an animated visualization?
I can create a video showing:
- Token-to-token attention weights
- Query/Key/Value projections
- Softmax normalization

Run: `/visualize attention`
```
