<h2 align="center">
  <strong>IPIGuard</strong>: A Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents
</h2>

<p align="center">
  <a href="https://greysahy.github.io/" target="_blank">Hengyu An</a><sup>1</sup> &nbsp; | &nbsp;
  <a href="https://jzhang538.github.io/jinghuaizhang/" target="_blank">Jinghuai Zhang</a><sup>2</sup> &nbsp; | &nbsp;
  <a href="https://tydusky.github.io/" target="_blank">Tianyu Du</a><sup>1</sup> &nbsp; | &nbsp;
  Chunyi Zhou<sup>1</sup> &nbsp; | &nbsp;
  Qingming Li<sup>1</sup> &nbsp; | &nbsp;
  <a href="https://tlin-taolin.github.io/" target="_blank">Tao Lin</a><sup>3</sup> &nbsp; | &nbsp;
  <a href="https://nesa.zju.edu.cn/index.html/" target="_blank">Shouling Ji</a><sup>1</sup>
</p>

<p align="center" style="font-size: 1rem;">
  <sup>1</sup> Zhejiang University &nbsp;&nbsp;
  <sup>2</sup> University of California, Los Angeles &nbsp;&nbsp;
  <sup>3</sup> Westlake University
</p>

<br>
<br>


<!-- <p align="center">
<a href="https://huggingface.co/sp12138sp/UCGM">:robot: Models</a> &ensp;
<a href="https://arxiv.org/abs/2505.07447">:page_facing_up: Paper</a> &ensp;
<a href="#label-bibliography">:label: BibTeX</a> &ensp;
  <br><br>
<a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-256x256" alt="PWC"></a> <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-512x512" alt="PWC"></a>
</p> -->

<div align='center'>
  <img src="assets/figure.png" width="75%">
  <p>
    <strong>Comparison of the traditional task execution paradigm (top) and our IPIGUARD (bottom)</strong>
  </p>
</div>


## 🔧 Installation

We recommend using Python ≥3.10.

```bash
# git clone
git clone https://github.com/Greysahy/ipiguard.git
cd ipiguard

# create conda environment
conda create -n ipiguard python=3.10
conda activate ipiguard

# install agentdojo
cd agentdojo
pip install -e .
```

---

## 🚀 Quick Start

### 1. Set Your OpenAI API Key in eval.sh

```bash
export OPENAI_API_KEY='<your_api_key>'
export OPENAI_BASE_URL='<your_base_url>'
```

### 2. Run Evaluation
We provide a ready-to-use shell script for running evaluations with specific agent, attack, and defense configurations. You can modify the arguments to evaluate different settings. 

To run the script:

```bash
bash eval.sh
```

---

## 📊 Results

| Attack            | Workspace       | Slack           | Travel          | Banking         | Overall         |
|------------------|------------------|------------------|------------------|------------------|------------------|
|                  | ASR↓ / UA↑       | ASR↓ / UA↑       | ASR↓ / UA↑       | ASR↓ / UA↑       | ASR↓ / UA↑       |
| Ignore Previous  | 0.00 / 68.33     | 0.00 / 59.05     | 0.00 / 62.86     | 2.78 / 49.31     | 0.64 / 61.21     |
| InjectAgent      | 0.42 / 67.92     | 0.95 / 63.81     | 0.00 / 65.00     | 0.00 / 47.92     | 0.32 / 61.84     |
| Tool Knowledge   | 0.00 / 69.58     | 1.90 / 59.05     | 0.00 / 59.29     | 2.78 / 47.92     | 0.95 / 60.57     |
| Important Instr. | 0.83 / 65.00     | 0.00 / 49.52     | 0.00 / 57.14     | 1.39 / 49.31     | 0.64 / 57.07     |
| Average          | 0.31 / 67.71     | 0.71 / 57.86     | 0.00 / 61.07     | 1.74 / 48.44     | 0.69 / 58.77     |

More results and ablations are available in the [paper](link_to_paper).

---

## 📌 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@tbd{an2025ipiguard,
  title     = {Paper Title},
  author    = {Author1 and Author2 and Author3},
  booktitle = {Proceedings of the ...},
  year      = {2025}
}
```
---


## 🏷️ License

Apache License 2.0 - See [LICENSE](LICENSE) for details.