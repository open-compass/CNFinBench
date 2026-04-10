#  CNFinBench: Evaluating Expertise, Autonomy, and Integrity in Finance

**Language**: [English](README.md) | [中文](README_CN.md)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.09506)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-OpenCompass-blue)](https://cnfinbench.opencompass.org.cn/home)

---

📹 **For clearer demo video, please click the link:** [Video](https://www.bilibili.com/video/BV1tCFKz7E5V)  

https://github.com/user-attachments/assets/c15e95fb-8081-474b-936a-3686ac312c62

## 📣 News & Announcements

📰 **Media Coverage**  
- 👉 [CNFinBench Released: A New Benchmark for Financial LLM Safety Evaluation](https://mp.weixin.qq.com/s/z427UB6r6QPNyAX_Pfc0JA)
- 👉 [From Capabilities to Agents: How CNFinBench Evaluates Compliance and Safety of Financial LLMs](https://mp.weixin.qq.com/s/5UUBlklvoB67nQYOIXee7Q)
- 👉 [Shanghai Government: Financial LLM Evaluation System 2.0 Released in Shanghai](https://www.shanghai.gov.cn/nw4411/20251229/3a87d71a2a104c9993164165a2e21c0a.html)
- 👉 [21st Century Business Herald: Industry Standards Upgraded! 2025 Financial LLM Evaluation System Officially Released in Shanghai](https://m.21jingji.com/article/20251229/herald/c4281b3b73c2b77940b8ab1010514875.html)
- 👉 [China News Service: 2025 Financial LLM Evaluation System Successfully Launched in Shanghai](https://www.sh.chinanews.com.cn/kjjy/2025-12-27/143671.shtml)
- 👉 [International Finance News: 2025 Financial LLM Evaluation System](https://www.ifnews.com/h5/news.html?aid=792908)
- 👉 [Xinhua Finance: 2025 Financial LLM Evaluation System](https://bm.cnfic.com.cn/sharing/share/articleDetail/429697918115848192/1)
- 👉 [China Securities Journal: 2025 Financial LLM Evaluation System](https://csapp1.cs.com.cn/hg/2025/12/28/detail_202512289338043.html)


📄 **Academic Release**  
- **Beyond Knowledge to Agency: Evaluating Expertise, Autonomy, and Integrity in Finance with CNFinBench**  
  🔗 https://arxiv.org/abs/2512.09506

🌐 **Online Leaderboard**  
- 🔥 Live leaderboard and model submission:   https://cnfinbench.opencompass.org.cn/home


---

## 📖 What is CNFinBench?


**CNFinBench** is a comprehensive benchmark for evaluating **large language models and agentic systems** in **high-stakes financial scenarios**.

Unlike traditional textbook-style financial QA benchmarks, CNFinBench targets **real-world deployment risks** introduced by *high-privilege financial agents*, and systematically evaluates models along three orthogonal axes:

- **Expertise** – professional financial knowledge and reasoning  
- **Autonomy** – multi-step planning, tool use, and agent execution  
- **Integrity** – safety, compliance, and robustness under adversarial interaction  

CNFinBench spans **29 fine-grained tasks**, grounded in certified regulatory corpora, real financial workflows, and multi-turn adversarial attack scenarios.

---

## 🧩 Task Taxonomy

CNFinBench decomposes financial intelligence into a three-dimensional evaluation space:

<div align="center">
<img align="center" src="assets/fig-task-taxonomy.png" width="80%"/>
</div>



### 📚Expertise – Financial Capability

- Financial Knowledge Mastery
- Complex Logic Composition
- Contextual Analysis Resilience

### ⚒️ Autonomy – Agentic Execution

- End-to-End Execution (Intent → Plan → Tool → Verification)
- Strategic Planning & Reasoning
- Meta-cognitive Reliability

### 🥷🏻 Integrity – Safety & Compliance

- Immediate Risk Interception
- Compliance Persistence
- Dynamic Adversarial Orchestration

---

## 🔐 Multi-turn Safety Evaluation & HICS


To quantify **behavioral compliance degradation**, CNFinBench introduces:

### **Harmful Instruction Compliance Score (HICS)**

- Multi-dimensional, severity-aware safety metric  
- Tracks violation escalation across dialogue rounds  
- Supports interpretable rule-level deduction logs  
- Reveals **collapse rhythms** under different attack strategies  

---

### 🏗 How is CNFinBench constructed?

CNFinBench adopts a multi-stage **data generation pipeline** that combines:

<div align="center">
<img align="center" src="assets/fig-data-gen.png" width="80%"/>
</div>

- **LLM-assisted synthesis** – for scalable question generation
- **Expert authoring & validation** – to ensure domain accuracy and risk coverage
- **Interaction & safety task design** – simulating trust boundaries and real agent execution chains
- **Task-aware rubric annotation** – enabling interpretable model evaluation across 3 axes

---

## 🏆 Leaderboard & Evaluation Platform


CNFinBench is deployed on a **fully automated evaluation platform** built on **OpenCompass**, supporting:

- Unified evaluation of **open-source & closed-source models**
- Task-aware rubrics with LLM-as-Judge protocols
- Real-time leaderboard updates
- Dynamic task and model integration

🔗 **Visit the leaderboard:**  
👉 https://cnfinbench.opencompass.org.cn/home

<div align="center">   <img src="./assets/image-20260131225448953.png" alt="Platform Overview" width="55%"> </div>

Below is a snapshot of the **current leaderboard** (updated in real time on the platform):

<div align="center">   <img src="./assets/image-20260131230239866.png" alt="Leaderboard Snapshot" width="55%"> </div>

---

## 📊 Benchmark Scale

- **29 subtasks** across Expertise / Autonomy / Integrity  
- **11,947 single-turn QA instances**  
- **321 multi-turn adversarial dialogues** (4 rounds each)  
- **22 evaluated models** (open-source, closed-source, finance-tuned)

---

## 💻 Code Usage

### Multi-turn Dialogue Evaluation

For **multi-turn adversarial dialogue evaluation**, please refer to the detailed guides in the `multi-turn` directory:

- 📖 **English Guide**: [multi-turn/README.md](multi-turn/README.md)
- 📖 **中文指南**: [multi-turn/README_CN.md](multi-turn/README_CN.md)

The multi-turn evaluation pipeline includes:
1. **Generate multi-turn dialogue tests** using scripts in `multi-turn/pred/`
2. **Merge output files** using `multi-turn/pred/merge.py`
3. **Evaluate results** using scripts in `multi-turn/judge/`

For detailed evaluation script documentation:
- 📖 **English**: [multi-turn/judge/README_EN.md](multi-turn/judge/README_EN.md)
- 📖 **中文**: [multi-turn/judge/README.md](multi-turn/judge/README.md)

### Quick Start

1. **Install dependencies**:
   ```bash
   cd multi-turn
   pip install -r requirements.txt
   ```

2. **Generate multi-turn dialogues**:
   ```bash
   cd pred
   python main.py --data-dir ../data --output-dir ../output --model-name your_model_name \
       --attack-api-key your_attack_api_key --attack-base-url your_attack_base_url \
       --attack-model-name your_attack_model --defense-api-key your_defense_api_key \
       --defense-base-url your_defense_base_url --defense-model-name your_defense_model
   ```

3. **Merge output files**:
   ```bash
   python merge.py --output-dir ../output
   ```

4. **Evaluate results**:
   ```bash
   cd ..
   python -m judge.evaluate --output-dir ./output \
       --judge-api-key your_judge_api_key --judge-base-url your_judge_base_url \
       --judge-model-name your_judge_model
   ```

For more detailed instructions and examples, please see the [multi-turn README](multi-turn/README.md).

---

## 📖 Citation

If you use CNFinBench, please cite:

```bibtex
@misc{ding2025cnfinbenchbenchmarksafetycompliance,
      title={CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance}, 
      author={Jinru Ding and Chao Ding and Wenrao Pang and Boyi Xiao and Zhiqiang Liu and Pengcheng Chen and Jiayuan Chen and Tiantian Yuan and Junming Guan and Yidong Jiang and Dawei Cheng and Jie Xu},
      year={2025},
      eprint={2512.09506},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2512.09506}, 
}
