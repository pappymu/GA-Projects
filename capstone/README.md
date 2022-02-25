# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 
# Test Everything, Carefully: Generating Questions from Text (T5 Transformer in Pytorch)
## GA DSI 26 Capstone Project

# Introduction

Education technology (edtech) is a crucial field that can help teachers with automate tasks so that they can focus on the more human aspects of their jobs, e.g. the personal interaction with students. With the maturing of Massive Open Online Courses (MOOCs), there is plenty of space for the automation of tasks like answer grading and setting test papers.

One of the many areas in edtech, question and answer generation can populate question banks for various stakeholders: for teachers to set exam papers, for students to test themselves, and for institutions to standardize their exam difficulty levels. With this, students and teachers can assess the skills attained by the learner and identify weak spots to focus on.  

There are various flavors of question generation, of which the one with the longest history is answer-aware question generation, where questions are generated from a given context paragraph and an inputted answer (e.g. [Zhou et al, 2017](https://arxiv.org/abs/1704.01792)). Work has also been done on generating multiple-choice questions with 3 distractors from a given text ([Vachev et al, 2022](https://arxiv.org/abs/2201.09012)). There is also answer-agnostic question generation, where a mask is given as the answer and a question-answer pair is generated from a given text ([Scialom et al, 2019](https://aclanthology.org/P19-1604.pdf)). There are of course many other directions in the literature (collected in this [github](https://github.com/teacherpeterpan/Question-Generation-Paper-List)).

The state-of-the-art models for text generation are transformer models, first introduced in the seminal paper [Attention Is All You Need](https://nlp.seas.harvard.edu/2018/04/03/attention.html). These deep learning models solved the problems of recurrent neural networks to provide robust text generation algorithms.

In this light, we aim to create a question and answer generator based on text given to the model. We will be mixing answer-agnostic and answer-aware methods, as we believe these will average out the problems in both methods while providing flexibility.

# Methodology

We will be training our models on the famous Stanford Question and Answering Dataset (SQuAD) dataset compiled by [Rajpurkar et al, 2016](https://arxiv.org/abs/1606.05250). We decided to use version 1.1 instead of 2.0 as the latter contains unanswerable questions, which is beyond the scope of our project.

We built a T5 Model ([Raffel et al, 2020](https://arxiv.org/abs/1910.10683)) in Pytorch, with the help of the hugging face library.

Lastly, we will be comparing the models based on their cross entropy loss, and the scoring results with BLEU and cosine similarity scores. In the process, we found out that none of the scoring methods are robust enough to ensure the quality of the questions, in accordance with [Nema et al, 2018](https://arxiv.org/abs/1808.10192), [Callison-Burch et al, 2006](https://aclanthology.org/E06-1032.pdf) and [Liu et al, 2016](https://arxiv.org/abs/1603.08023v1) which showed that "these metrics correlate very weakly or not at all with human judgements of the response quality".

# Scores
## Cross-entropy: 1.23

## Answer-aware
- average BLEU Score: 0.102
- average Cosine Similarity Score: 0.856

## Answer-agnostic
- average BLEU Score: 0.101
- average Cosine Similarity Score: 0.841
