# mathopt
直線探索とかNesterovとかの実装

# Usage
## 最急降下法

次のように使う。ステップ幅の決め方は，目的関数が $L$ -平滑だと $\frac{1}{L}$ とかにする。

```python
from mathopt.algs import SteepestDescent

stp = SteepestDescent(f, df, stepsize, n_iter)
stp.run(x_init, callbacks=[...])
```

デフォルトでは backtracking をしていないが，引数に `armijo = True` を指定すると Armijo Rule を満たすステップ幅を探す。
このときは `stepsize = 1` とかでいい。

```python
from mathopt.algs import SteepestDescent

stp = SteepestDescent(f, df, stepsize, n_iter, armijo = True)
stp.run(x_init, callbacks=[...])
```

## Nesterov

使い方は同じ。

```python
from mathopt.algs import Nesterov

nst = Nesterov(f, df, stepsize, n_iter)
nst.run(x_init, callbacks=[...])
```

# Callbackについて

`mathopt.callback.core.Callback` を継承したクラスを `run` メソッドに指定することで各イテレーション時に実行する関数を実装できる。

実装済みのもの：

## ロギング
各イテレーションでの目的関数の値を出力する(デバッグ用)。

```python
from mathopt.callbacks import Logging

logging = Logging()

stp = SteepestDescent(f, df, stepsize, n_iter)
stp.run(x_init, callbacks=[logging]) # <- ココ
```

そうすると，

```bash
1-th iteration: Objective 0.34725218680384945
2-th iteration: Objective 0.13945951227613315
3-th iteration: Objective 0.058190904435611465
4-th iteration: Objective 0.025401816995384803
...
```

## レコード
各イテレーションでの目的関数の値をリストにもつ。

```python
from mathopt.callbacks import Record

recorder = Record(result=[])

stp = SteepestDescent(f, df, stepsize, n_iter)
stp.run(x_init, callbacks=[recorder]) # <- ココ

print(recorder.result) # 目的関数の各イテレーションでの値
```

## 軌跡
各イテレーションでの変数の値をリストにもつ。

Trajectory の引数の `traj` には shape が $(N, 1)$ の形に変形した初期点を指定することを想定されている。

```python
from mathopt.callbacks import Trajectory

traj = Trajectory(traj = x_init)

stp = SteepestDescent(f, df, stepsize, n_iter)
stp.run(x_init, callbacks=[traj]) # <- ココ

print(traj.traj) # Shape (N, n_iter) の軌跡
```
