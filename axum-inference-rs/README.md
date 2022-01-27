# Rust axum based python inference service

## Getting Started

```bash
cargu build --release
```


```bash
chmod +x ./target/release/axum-inference-rs
./target/release/axum-inference-rs
```

## Test

```bash
hey -n 200 -c 1 -m POST -H 'Content-Type: application/json' -d '{	"inputs": "I love you. I like you. I am your friend."}' http://127.0.0.1:3000/predict
```

Runs for `-c 1`

```
ummary:
  Total:        6.8468 secs
  Slowest:      0.0543 secs
  Fastest:      0.0311 secs
  Average:      0.0342 secs
  Requests/sec: 29.2109
  
  Total data:   8000 bytes
  Size/request: 40 bytes

Response time histogram:
  0.031 [1]     |
  0.033 [145]   |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.036 [5]     |■
  0.038 [13]    |■■■■
  0.040 [16]    |■■■■
  0.043 [4]     |■
  0.045 [4]     |■
  0.047 [2]     |■
  0.050 [3]     |■
  0.052 [1]     |
  0.054 [6]     |■■
```


Runs for `-c 2`

```bash
Summary:
  Total:        5.5871 secs
  Slowest:      0.0655 secs
  Fastest:      0.0483 secs
  Average:      0.0558 secs
  Requests/sec: 35.7965
  
  Total data:   8000 bytes
  Size/request: 40 bytes

Response time histogram:
  0.048 [1]     |■
  0.050 [0]     |
  0.052 [1]     |■
  0.053 [25]    |■■■■■■■■■■■■■■■
  0.055 [62]    |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.057 [67]    |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.059 [20]    |■■■■■■■■■■■■
  0.060 [14]    |■■■■■■■■
  0.062 [6]     |■■■■
  0.064 [2]     |■
  0.066 [2]     |■
```