---
layout: post
title: Poisson Kernel on the Upper Half-Space Integrates to 1 on the Boundary
date: 2021-08-16 11:12:00+0800
description: n维空间上半空间泊松核 (Poisson Kernel) 在边界积分为1的一个证明
tags: math
# categories: lecture notes
related_posts: true
---
在 Evans PDE 的 P38 关于 Laplace 方程解的推导中需用到如下积分:

$$\int_{\partial \mathbb{R}^{n}_{+}} K(x,y)\, \mathrm{d}y = 1,$$

其中

$$K(x, y):=\frac{2 x_{n}}{n \alpha(n)} \frac{1}{|x-y|^{n}} \quad(x \in \mathbb{R}_{+}^{n}, y \in \partial \mathbb{R}_{+}^{n})$$

为 $$n$$ 维空间上的泊松核 (Poisson Kernel), $$\mathbb{R}^{n}_{+}=\{(y_1,y_2,\cdots,y_n)\mid y_n > 0\}$$ 表示 $$n$$ 维空间的上半空间, $$\partial \mathbb{R}^{n}_{+}=\mathbb{R}^{n-1}$$ 为 $$\mathbb{R}^{n}_{+}$$ 的边界,

$$\alpha(n)=\frac{\pi^{n/2}}{\Gamma(\frac{n}{2}+1)}$$

为 $$\mathbb{R}^{n}$$ 中单位球的体积. 利用 $$\partial \mathbb{R}^{n}_{+}$$ 与 $$\mathbb{R}^{n-1}$$ 的等价性可得

$$\begin{aligned}
\int_{\partial \mathbb{R}_+^n} K(x,y) \,\mathrm{d}y &= \int_{\mathbb{R}^{n-1}} K(x,y) \,\mathrm{d}y \\&= \int_{\mathbb{R}^{n-1}} \frac{2x_n}{n \alpha(n)} \frac{1}{|x-y|^n} \,\mathrm{d}y \\
&=  \frac{2x_n}{n \alpha(n)} \int_{\mathbb{R}^{n-1}}\frac{1}{(x_{n}^2+y_{1}^2+...+y_{n-1}^2)^{\frac{n}{2}}} \,\mathrm{d}y \\&= \frac{2x_n}{n \alpha(n)} \int_{\mathbb{R}^{n-1}}\frac{1}{(x_{n}^2+|y|^2)^{\frac{n}{2}}} \,\mathrm{d}y\\
&= \frac{2}{n \alpha(n)x_n^{n-1}} \int_{\mathbb{R}^{n-1}}\frac{1}{(1+|y/x_n|^2)^{\frac{n}{2}}} \,\mathrm{d}y
\end{aligned}$$

做变量代换 $$z=y/x_n$$ 得

$$\int_{\partial \mathbb{R}_+^n} K(x,y) \,\mathrm{d}y = \frac{2}{n \alpha(n)} \int_{\mathbb{R}^{n-1}}\frac{1}{(1+|z|^2)^{\frac{n}{2}}} \,\mathrm{d}z.$$

借助 $$n-1$$ 维空间中的极坐标变换, 上式可以写为

$$\begin{aligned}
   \frac{2}{n \alpha(n)} \int_{\mathbb{R}^{n-1}}\frac{1}{(1+|z|^2)^{\frac{n}{2}}} \,\mathrm{d}z &= \frac{2}{n \alpha(n)} \int_{0}^{\infty}\int_{\partial B(0,r)}\frac{1}{(1+r^2)^{\frac{n}{2}}}\,\mathrm{d}S  \,\mathrm{d}r \\
&= \frac{2(n-1)\alpha(n-1)}{n \alpha(n)} \int_{0}^{\infty}\frac{r^{n-2}}{(1+r^2)^{\frac{n}{2}}} \,\mathrm{d}r,
\end{aligned}$$

其中 $$B(0,r)=\{x\in \mathbb{R}^{n-1}\mid \|x\|^2<r\}$$ 是 $$n-1$$ 维空间中以原点为球心, $$r$$ 为半径的开球. 第二个等号用到了 $$n-1$$ 维空间中球的表面积公式

$$\int_{\partial B(0,r)} 1 \,\mathrm{d}S = (n-1)\alpha(n-1)r^{n-2}.$$

记 $$S(n)=n\alpha(n)$$, 它实际上等于 $$n$$ 维空间中单位球的表面积, 利用 $$\Gamma$$ 函数的性质

$$\begin{aligned}
\Gamma(z+1) &=\int_{0}^{\infty} x^{z} e^{-x} \,\mathrm{d} x \\
&=\left[-x^{z} e^{-x}\right]_{0}^{\infty}+\int_{0}^{\infty} z x^{z-1} e^{-x} \,\mathrm{d} x \\
&=\lim_{x \rightarrow \infty}\left(-x^{z} e^{-x}\right)-\left(-0^{z} e^{-0}\right)+z \int_{0}^{\infty} x^{z-1} e^{-x} \,\mathrm{d} x\\
&=z\Gamma(z)
\end{aligned}$$

知 $$\Gamma(\frac{n}{2}+1)=\frac{n}{2}\Gamma(\frac{n}{2})$$, 于是

$$S(n)=n\alpha(n)=\frac{n\pi^{n/2}}{\Gamma(\frac{n}{2}+1)}=\frac{2\pi^{n/2}}{\Gamma(\frac{n}{2})}.$$

最终我们得到

$$\begin{aligned}
    \frac{2(n-1)\alpha(n-1)}{n \alpha(n)} \int_{0}^{\infty}\frac{r^{n-2}}{(1+r^2)^{\frac{n}{2}}} \,\mathrm{d}r = \frac{2S(n-1)}{S(n)} \int_{0}^{\infty}\frac{r^{n-2}}{(1+r^2)^{\frac{n}{2}}} \,\mathrm{d}r \\
    = \frac{2\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})\sqrt{\pi}}\int_{0}^{\infty}\frac{r^{n-2}}{(1+r^2)^{\frac{n}{2}}}\,\mathrm{d}r.
\end{aligned}$$


> 于是只需验证
> 
> $$\int_{0}^{\infty}\frac{r^{n-2}}{(1+r^{2})^{\frac{n}{2}}}\,\mathrm{d}r=\frac{\Gamma(\frac{n-1}{2})\sqrt{\pi}}{2\Gamma(\frac{n}{2})}.$$
> 
> 我们对 $$n$$ 归纳, 当 $$n=2$$ 时
> 
> $$\begin{aligned}
    \int_{0}^{\infty}\frac{1}{1+r^2}\,\mathrm{d}r&=\int_{0}^{\frac{\pi}{2}}\frac{1}{\cos^{2}r(1+\tan^{2}r)}\,\mathrm{d}r\\
    &=\int_{0}^{\frac{\pi}{2}}\frac{\cos^2r}{\cos^2r}\,\mathrm{d}r=\frac{\pi}{2}.
\end{aligned}$$
> 
> $$n\geq 3$$时, 由分部积分可知
> 
> $$\begin{aligned}
    \int_{0}^{\infty}\frac{r^{n-2}}{(1+r^{2})^{\frac{n}{2}}}\,\mathrm{d}r
    &=-\frac{1}{n-2}\int_{0}^{\infty}r^{n-3}\,\mathrm{d}\frac{1}{(1+r^2)^{\frac{n-2}{2}}}\\
    &=\underbrace{\left[-\frac{1}{n-2}\frac{1}{(1+r^2)^{\frac{n-2}{2}}}r^{n-3}\right]_{0}^{\infty}}_{=0}+\frac{n-3}{n-2}\int_{0}^{\infty}\frac{r^{n-4}}{(1+r^{2})^{\frac{n-2}{2}}}\,\mathrm{d}r.
\end{aligned}$$
> 
> 由归纳法的假设
> 
> $$\begin{aligned}
    \int_{0}^{\infty}\frac{r^{n-4}}{(1+r^{2})^{\frac{n-2}{2}}}\,\mathrm{d}r=\frac{\Gamma(\frac{n-3}{2})\sqrt{\pi}}{2\Gamma(\frac{n-2}{2})}
\end{aligned}$$
> 
> 及 $$\Gamma$$ 函数的性质 $$z\Gamma(z)=\Gamma(z+1)$$ 得
> 
> $$\begin{aligned}
    \int_{0}^{\infty}\frac{r^{n-2}}{(1+r^{2})^{\frac{n}{2}}}\,\mathrm{d}r&=\frac{n-3}{n-2}\int_{0}^{\infty}\frac{r^{n-4}}{(1+r^{2})^{\frac{n-2}{2}}}\,\mathrm{d}r\\
    &=\frac{\frac{n-3}{2}}{\frac{n-2}{2}}\frac{\Gamma(\frac{n-3}{2})\sqrt{\pi}}{2\Gamma(\frac{n-2}{2})}\\
&=\frac{\Gamma(\frac{n-1}{2})\sqrt{\pi}}{2\Gamma(\frac{n}{2})},
\end{aligned}$$
> 
> 故结论成立.

