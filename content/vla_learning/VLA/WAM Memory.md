MemoryWAM: Efficient World Action Modeling with Persistent Memory

Background:
```
Prior WAMs typically face a memory-efficiency tradeoff: slidding window is efficient but forgets long-range context, while adding context length will increase computational cost.(full-history KV cache preserves context but scales linearly with trajectory length N)
```

MemoryWAM 提出了hybrid memory， 即长短期记忆
![[Pasted image 20261104134519.png]]
