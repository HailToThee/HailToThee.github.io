		
以 Zero-WAM / LingBot-VA 为例：

$$
\text{Demo / Instruction} + V_{\le t}
\rightarrow
\hat V_{t+1}
\rightarrow
A_t
$$

即：
1. 根据 Human Demo、Instruction 和当前 Robot Observation 预测 Future Robot Video；
2. 根据 Future Visual State 预测 Robot Action。

Fast_WAM 提出了不必完整地预测整个 $V_{t+1}$, OpenWAM也讲到了这一点
因为生成 $V_{t+1}$ 需要建模大量可能与 Action 无关的细节信息，例如：
	- Background
- Texture
- Appearance details
- Pixel-level details

需要的信息可能只是：

> **当前 Visual State 接下来应该朝什么方向变化？**


$$
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
A_t
$$

不显式生成 $\hat z_{t+1}$，而是预测从当前状态指向 Future State 的方向：
通过
$$
V_t
\rightarrow
d_t
\rightarrow
A_t
$$


如何得到dt?
```
先使用最简单的 Latent Displacement：

d_t = z_{t+1} - z_t
	
训练 Visual Dynamics Predictor：

\hat d_t = D_\phi(z_t,H)

minimize MSE loss or KL divergence
```

```
上述的可能存在一个问题，即dt仅凭借z_t z_t+1能否对应正确的time step？
对于d_t来说，应该是通过前面所有的o_t得到的，因此需要维护一个状态量s_t
s_t = mamba(z_1:t)
由(H, s_t, z_t+1)  --> d_t
```
Training

训练阶段仍然可以保留 Future Prediction 作为 Auxiliary Supervision, 因为仍然可能走捷径
看看使用预测出来的方向能否指导恢复？
Future Loss：

$$
L_{\mathrm{future}}
=
d(\hat z_{t+1},z_{t+1})
$$

结构：

```text
Human Demo H
      +
Current Robot Video V_t
      ↓
Visual Dynamics Predictor
      ↓
    d_t
   /   \
  /     \
 ↓       ↓
V_t+1   Action
 ↓       ↓
L_future L_action
```

Inference

推理时删除 Future Reconstruction Branch， 不显式地生成V_t+1


>human video In context learning的作用？

由于human video和robot video之间存在gap, 因此不能直接 human video --> action
而是需要加一个连接使得
(h, v_t) --> dt
(v_t, dt) --> At

如何设计？
1. 整体直接输入到input端，然后crossattn得到一个visual latent
2. 可以参照Bahavior Prompting Policy？也是根据当前的o_t以及动作状态，取argmax(attn(cur, video chunks))得到当前对应的human帧的chunk?再根据这个chunk和下一个chunk作为input，进行输入来指导dt的生成？然而存在问题，第一点是数据问题，Bahavior Prompting Policy的训练数据中有直接对应的当前动作。第二点是embodiment gap的问题，这个能否在训练中缓解？
3. 如果使用了状态量s_t， 对于human demo，能否也通过mamba编码成N个embedding，再使用bahavior prompting policy进行状态的匹配？这个是否会没有那么大的embodiment gap
df

如何包含past的trajectory?
from zerowam... bahavior prompting policy, 如果不包含trajectory的话，当前帧能够提供fairly多的信息，因此未必会考虑添加的额外部分。也就是直接input进去作为context可能并不work

如何把这个trajectory输入到context里面？强行关注trajectory信息。
with post training or without post training
if with post training

1. follow zerowam等通过添加对未来帧的预测mse loss显式将video context引入进来，和上面介绍的类似
2. 实现一种对context压缩的方法，通过post training让模型能够理解包含trajectory等信息。
3. geometric transfer, trajectory may also be presented  in hyper latent space? 相关论文的调研
2,3 很可能可以融合

long-wam
in context learning for robots调研1
记忆和incontext十分类似




1.Mamba...

1. 使用预训练 Wan VAE 将 Human Demonstration Video 编码为 Visual Tokens，再通过 Mamba 建模时序信息，取最后一个 Visual Token 的 Hidden State 作为 Context Latent，输入 VLA 指导动作生成。
    
2. Mamba 需要学习如何保留任务相关的时序信息，可能需要额外 Pretraining，与 Minimal Training 的目标存在冲突。
    
3. 能否不使用单个 Token，而是将 Video 压缩为 \(N\) 个具有时序信息的 Embeddings，再根据当前 Robot Observation \(o_t\)，通过 Attention 匹配对应的 Human Demonstration 阶段，提取相关 Context 指导 Action Generation？
![[aem(mamba)-arc.png]]


![[Pasted image 20261104134802.png]]
选任务长记忆

2.latent 

如何设计？

1. 分别使用预训练 Wan VAE 和 DINOv3  Phoenix对 Human Demonstration Video 进行编码，得到每一帧对应的 Latent，观察其在 Latent Space 中的演化轨迹。
    
2. 由于 Wan 和 DINO 的特征都可以进行归一化，能否比较两者的 Latent Trajectory 是否具有相似的几何性质？例如相邻帧变化方向、轨迹曲率、局部切空间以及状态转移趋势。
    
3. 如果两种不同预训练目标的模型得到相似的 Latent Trajectory，是否意味着其中存在相对稳定的任务动态信息？能否利用这种动态结构提取有效的 Video Context，而不需要额外预训练？
    
4. 进一步比较相同任务下 Human Video 和 Robot Video 的 Latent Trajectory，观察其运动方向、局部几何结构是否相似，探索能否通过共享的动态结构缓解 Embodiment Gap。


数据集:Human And Robot 按照帧对应
按照每秒10frams提取特征
各个特征形状:
- wan: [T' * H‘ * W', C]
- Dinov3 CLS: [T', D]
- Dinov3 Patch:[T' * H' * W', D]
- Phoenix-CS-VAE: [T' * H' * W', C]
- Phoenix-Joint96: [T' * H' * 2W', C] (Phoenix VAE+ CS-VAE concated)
空间结构相似(State RSA):分别计算Human 和 Robot 轨迹内部任意两个时刻的距离，构造距离矩阵，再计算两个距离矩阵上三角元素的 Pearson 相关系数
- $D_{ij}=|z_i-z_j|_2$
- $\mathrm{R}=\operatorname{Corr}(D_H,D_R)$

| 模型      | 表示      | L2 State R | Raw State R |
| ------- | ------- | ---------- | ----------- |
| Wan     | Latent  | 0.6383     | 0.5872      |
| DINOv3  | CLS     | 0.5832     | 0.5839      |
| DINOv3  | Patch   | 0.6038     | 0.6070      |
| Phoenix | CS-VAE  | **0.8408** | **0.8374**  |
| Phoenix | Joint96 | 0.7717     | 0.7321      |
- raw state: 使用原特征
- l2 state: 归一化后的特征
空间相似度来说 Phoenix > wan > dino
Phoenix的CS-VAE能够达到84%， 猜测可能是使用了robot数据进行pretrain的原因:contains robot demonstrations, egocentric human videos

空间变化:
- $\delta z_t = z_{t+1} - z_{t}$
- $r_{mag} = Corr(\|\delta z_{t}\|_2)$

$$\Delta z_t=z_{t+1}-z_t,\qquad  
d_t=\frac{\Delta z_t}{|\Delta z_t|}$$
$$\kappa_t=  
\frac{|d_{t+1}-d_t|}  
{\frac12(|\Delta z_t|+|\Delta z_{t+1}|)+\epsilon}$$
$$\text{Curvature Correlation}  
=\rho(\kappa^H,\kappa^R)$$

| 模型              | 幅度相关   | 曲率相关   | 对应方向余弦  | 内部方向余弦 Pearson |
| --------------- | ------ | ------ | ------- | -------------- |
| Wan             | 0.3700 | 0.3531 | 0.0731  | 0.0604         |
| DINOv3 CLS      | 0.0687 | 0.1000 | -0.0007 | 0.0737         |
| DINOv3 Patch    | 0.0898 | 0.1205 | -0.0006 | 0.0668         |
| Phoenix CS-VAE  | 0.8734 | 0.5408 | 0.0935  | 0.1510         |
| Phoenix Joint96 | 0.4840 | 0.4369 | 0.0878  | 0.0882         |
|                 |        |        |         |                |
相邻帧的空间变化的几个量也都是Phonix > wan > dino
其中幅度Phoenix达到了87.34, 但相应的方向余弦最大也只有0.09, 内部方向的二者相似度也不高
gap主要就是集中在了方向上。这里的幅度相关和上面的空间相似性大体上是一致的

测地线的一些结果:
公式
$x_t = \frac{z_t}{\|z_t\|}$
$u_t = {x_{t+1}-(x_{t+1}\top x_t)x_t}.norm$
$G_{ij} = \langle u_i, u_j\rangle$
$$\rho\left(  
\operatorname{upper}(G^H),  
\operatorname{upper}(G^R)  
\right)$$
对应方向余弦:
$G'_{ii} = \langle u_{Hi}, u_{Ri} \rangle$

	角速度:
	$$\theta_t=  
	\arccos\left(  
	\operatorname{clip}(x_t^\top x_{t+1},-1,1)  
	\right)$$
	$$\omega_t=  
	\frac{\theta_t}{\tau_{t+1}-\tau_t}$$
	$$\rho(\omega^H,\omega^R)$$
	
	| 模型              | Direction R | 对应方向余弦 | Angular Speed Correlation |
	| --------------- | ----------- | ------ | ------------------------- |
	| Wan             | 0.0571      | 0.0758 | 0.4538                    |
	| DINOv3 CLS      | 0.0721      | 0.0028 | 0.0625                    |
	| DINOv3 Patch    | 0.0668      | 0.0014 | 0.0745                    |
	| Phoenix CS-VAE  | 0.1211      | 0.0858 | 0.9351                    |
	| Phoenix Joint96 | 0.0805      | 0.0867 | 0.6300                    |
	
	|模型|完整序列|删除 1 个原生步|删除 4 个原生步|
	|---|---|---|---|
	|Wan|0.4538|0.2599|0.2820|
	|DINOv3 CLS|0.0625|0.0599|0.0531|
	|Phoenix CS-VAE|0.9351|0.3274|0.1560|
	|Phoenix Joint96|0.6300|0.3215|0.2850|

不一定真的要逐帧对应，而是要通过incontext让模型知道这个任务大概长什么样子
这个需要寄希望于两件事，第一件是pretrain
第二是视频迁移那边是怎么做的，能够让human到robot，从而经过robot训练的模型能够更好地理解，是再造数据吗


对于Vit来说，patchify后经过embedding得到的矩阵是一个二维的矩阵，形状为$[(H*W/size^2), C]$
1. CLIP can use 1 dim feature to classify: that's because it has added a [cls] token(like Bert) to extract the global feature of an image.
2. Dino use the same arc of ViT and [cls] token, so one can either use 1 dim feature or two dim feature.
3. Wan-VAE's output shape is $[B, T', H', W', C]$ and after being patchified $[B, T'*H'*W'/n^2, C]$, the same shape as ViT output. And another key is that Wan-VAE does not used transformer, so local shift can be observed.
 

### 1. Problem Formulation

Let the joint loss function be defined as:
$$
\mathcal{L} = \lambda_1 \mathcal{L}_1(f(x), v) + \lambda_2 \mathcal{L}_2(g(f(x), x), a)
$$
where:
- $x = [c, o_t, s_t]$ is the input observation (context, image, state).
- $f(\cdot)$ is the DiT architecture for video feature extraction.
- $g(\cdot, \cdot)$ is the action prediction architecture.
- $v$ and $a$ are the ground-truth video and action targets, respectively.
- $\lambda_1, \lambda_2$ are balancing coefficients.

To minimize the loss, particularly the action term $\mathcal{L}_2$, we introduce a perturbation $\delta$. We analyze two scenarios: adding $\delta$ to the feature representation $f(x)$ and adding $\delta$ directly to the input $x$.

---

### 2. Scenario A: Perturbation on Feature Representation $f(x)$

We define the perturbed loss as:
$$
\mathcal{L}'_A = \lambda_1 \mathcal{L}_1(f(x)+\delta, v) + \lambda_2 \mathcal{L}_2(g(f(x)+\delta, x), a)
$$

**First-order Taylor Expansion:**
$$
\begin{align}
\mathcal{L}'_A &\approx \lambda_1 \left[ \mathcal{L}_1(f(x), v) + \nabla_{f} \mathcal{L}_1^{\top} \delta \right] 
   + \lambda_2 \left[ \mathcal{L}_2(g(f(x), x), a) + \nabla_{f} \mathcal{L}_2^{\top} \delta \right] \\
&= \mathcal{L}(x) + \underbrace{\left[ \lambda_1 \nabla_{f} \mathcal{L}_1 + \lambda_2 \nabla_{f} \mathcal{L}_2 \right]^{\top}}_{\Delta_f^{\top}} \delta
\end{align}
$$
where $\nabla_{f} \mathcal{L}_2 = \left(\frac{\partial g}{\partial f}\right)^{\top} \nabla_{g} \mathcal{L}_2$.

**Optimal Perturbation:**
To minimize $\mathcal{L}'_A$, we choose $\delta$ opposite to the gradient direction, constrained by norm $\epsilon$:
$$
\|\delta\|_2 \leq \epsilon \quad \Rightarrow \quad \delta^*_A = -\epsilon \cdot \frac{\Delta_f}{\|\Delta_f\|_2 + \xi}
$$

**Maximal Loss Reduction:**
$$
\min_{\|\delta\|_2 \leq \epsilon} \mathcal{L}'_A \approx \mathcal{L}(x) - \epsilon \left\| \lambda_1 \nabla_{f} \mathcal{L}_1 + \lambda_2 \nabla_{f} \mathcal{L}_2 \right\|_2
$$

---

### 3. Scenario B: Perturbation on Input Observation $x$

We define the perturbed loss as:
$$
\mathcal{L}'_B = \lambda_1 \mathcal{L}_1(f(x+\delta), v) + \lambda_2 \mathcal{L}_2(g(f(x+\delta), x+\delta), a)
$$

**First-order Taylor Expansion:**
Using the chain rule, the gradient with respect to $x$ involves both direct and indirect paths through $f$ and $g$:
$$\begin{align}
\mathcal{L}'_B &\approx \lambda_1 \left[ \mathcal{L}_1(f(x), v) + \nabla_{f} \mathcal{L}_1^{\top} \frac{\partial f}{\partial x} \delta \right] \\
&\quad + \lambda_2 \left[ \mathcal{L}_2(g(f(x), x), a) + \nabla_{g} \mathcal{L}_2^{\top} \left( \frac{\partial g}{\partial f} \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x} \right) \delta \right] \\
&= \mathcal{L}(x) + \underbrace{\left[ \lambda_1 \left(\frac{\partial f}{\partial x}\right)^{\top} \nabla_{f} \mathcal{L}_1 + \lambda_2 \left( \left(\frac{\partial f}{\partial x}\right)^{\top} \left(\frac{\partial g}{\partial f}\right)^{\top} \nabla_{g} \mathcal{L}_2 + \left(\frac{\partial g}{\partial x}\right)^{\top} \nabla_{g} \mathcal{L}_2 \right) \right]^{\top}}_{\Delta_x^{\top}} \delta
\end{align}
$$
**Optimal Perturbation:**
Similarly, we constrain the input perturbation:
$$
\|\delta\|_2 \leq \epsilon \quad \Rightarrow \quad \delta^*_B = -\epsilon \cdot \frac{\Delta_x}{\|\Delta_x\|_2 + \xi}
$$

**Maximal Loss Reduction:**
$$
\min_{\|\delta\|_2 \leq \epsilon} \mathcal{L}'_B \approx \mathcal{L}(x) - \epsilon \left\| \lambda_1 \left(\frac{\partial f}{\partial x}\right)^{\top} \nabla_{f} \mathcal{L}_1 + \lambda_2 \left( \left(\frac{\partial f}{\partial x}\right)^{\top} \left(\frac{\partial g}{\partial f}\right)^{\top} \nabla_{g} \mathcal{L}_2 + \left(\frac{\partial g}{\partial x}\right)^{\top} \nabla_{g} \mathcal{L}_2 \right) \right\|_2
$$

---

### 4. Discussion

- **Feature-level Perturbation ($\delta$ on $f(x)$):** Directly optimizes the semantic representation used by the action head. It isolates the impact on $\mathcal{L}_2$ while keeping the raw input $x$ intact, preserving video fidelity $\mathcal{L}_1$ more effectively if $\nabla_f \mathcal{L}_1$ is small.
- **Input-level Perturbation ($\delta$ on $x$):** Affects both the feature extractor $f$ and the action head $g$ directly. The gradient $\Delta_x$ is modulated by the Jacobian $\frac{\partial f}{\partial x}$, meaning the perturbation's effectiveness depends on the local sensitivity of the video encoder. This approach may lead to larger changes in $\mathcal{L}_1$ due to direct modification of pixel/state values.
- **Trade-off Parameter $\epsilon$:** In both scenarios, $\epsilon$ controls the magnitude of improvement in action prediction at the cost of potential degradation in video generation or input consistency. $\epsilon \to 0$ recovers the original joint training objective.

想要的效果：在初始位置不太一样的情况下，能否做到同样的事情
	比如说拿手机，我们给的演示是在左手边，实际情况是在右手边
模型能否知道这件事情是要干什么，知道要干的是同一件事情。

现在能够通过一些工具拿到动作，以及相对的动作变换。这个可以用来试着验证一下在初始位置大致相似，运动轨迹大致相似的情况下能否跑出来。