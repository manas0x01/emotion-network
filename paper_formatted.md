EmoSphere-XAI: A Dual Transformer Explainable Intelligence Framework for Sustainable Open-Vocabulary Emotion Reasoning from Text

Manas Saxena2School of Computer Science & EngineeringGalgotias UniversityGreater Noida, Indiamanassaxena8954@gmail.com

Shubhranshu Shekhar Dash3

School of Computer Science & EngineeringGalgotias UniversityGreater Noida, India

Shubhranshudash24@gmail.com

Anjali Yadav2School of Computer Science & EngineeringGalgotias UniversityGreater Noida, Indiaanjaliydv167@gmail.com

Prof. Rajeev Kumar Singh4

EE Department

Indian Institute of Technology (Banaras Hindu University)

Uttar Pradesh, India

rksingh.eee@iitbhu.ac.in

Abstract— Emotion detection from text has emerged as a key research area in Natural Language Processing (NLP) because of its applications in intelligent human–computer interaction, affective computing, contextual sentiment analysis, and sustainable human-centric artificial intelligence. There is an urgent need to develop computationally efficient and transparent emotion recognition systems to promote responsible AI while reducing computational overhead and energy consumption. This paper proposes SustainSphere-XAI, a sustainable and explainable dual-transformer framework for fine-grained open-vocabulary emotion reasoning from textual data. The proposed framework combines the state-of-the-art ModernBERT-Large-Go-Emotions model with a lightweight fine-tuned DistilBERT architecture to achieve accurate contextual emotion understanding while providing a computational efficiency suitable for real-time deployment. The framework utilizes semantic embedding expansion, occlusion-based Explainable Artificial Intelligence (XAI), and entropy-based uncertainty estimation for transparent and trustworthy decision-making for enhanced interpretability and trustworthiness. The experimental results show that our method has better contextual emotion understanding, high-confidence emotion prediction, lower computational complexity and efficient inference performance. The proposed framework integrates explainability and lightweight transformer architectures to promote sustainable AI through energy-efficient inference, responsible resource utilization, digital well-being, and mental health support. This framework is a step towards environmentally sustainable and socially responsible intelligent systems, aligned with the broader goals of Green AI and the United Nations Sustainable Development Goals (SDGs), such as good health and well-being, industry innovation, and responsible development of technology.

Keywords- Emotion Detection, Explainable AI, NLP, ModernBERT, DistilBERT, Sustainable AI, Human-Centric AI, Responsible AI, Semantic Analysis, Text Classification.

INTRODUCTION

The Emotion detection from textual information has become a fast-growing research area in the fields of Natural Language Processing (NLP), Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), affective computing and sustainable human-centric AI systems. The explosive growth of social media platforms, online communication systems, digital forums, and conversational agents has made it increasingly important to understand the emotions embedded in textual content for intelligent human-computer interaction, sentiment understanding, mental health analysis, recommendation systems, CNN, Bi-GRU [1], and context-aware AI applications. Emotion detection focuses on identifying fine-grained emotional states such as joy, anger, sadness, fear, surprise, admiration, disappointment, and contextual affective expressions, unlike conventional sentiment analysis that mainly categorizes text into positive, negative, or neutral sentiments. Furthermore, sustainable AI-driven emotion analysis systems can contribute toward digital well-being, emotionally adaptive educational technologies, responsible social media monitoring, and transparent intelligent systems that support ethical and socially beneficial computing environments.

Previous computational methods for emotion detection were mostly lexicon-based methods, rule-based systems, handcrafted linguistic features, and traditional machine learning algorithms such as Support Vector Machines (SVM), Naïve Bayes, Deep Learning Neural Networks [2], and Decision Trees. These methods achieved moderate classification performance but failed to effectively capture the semantic context, long-range linguistic dependencies, implicit emotional cues, sarcasm, and subtle emotional reasoning in natural language text. Recent advances in deep learning and transformer-based architectures have greatly enhanced the capability of NLP systems by enabling contextual representation learning through attention mechanisms and large-scale pre-training strategies. In addition, the growing emphasis on sustainable and responsible AI has increased the demand for computationally efficient, explainable, and energy-aware transformer frameworks capable of delivering transparent emotion inference while supporting scalable real-time deployment in intelligent systems.

Existing Research and Challenges :

Recent studies explored transformer-based emotion recognition systems like BERT, RoBERTa, ModernBERT and DistilBERT for textual emotion analysis. Research studies such as towards Text-based Emotion Detection: A Survey and Possible Improvements, Text-Based Emotion Recognition Using Deep Learning and Machine Learning Approach, and Computational Approaches for Emotion Detection in Text [3] have highlighted the increasing importance of contextual language understanding and deep neural architectures in affective computing systems. Furthermore, a number of studies have highlighted the importance of dealing with class imbalance by employing weighted loss functions, bidirectional contextual processing, and regularization techniques such as dropout layers to improve classification robustness and generalization performance. Recent advances on transfer learning have also adopted lexicon-based approach in sentiment analysis [4] tasks before emotion classification can effectively enhance affective understanding and knowledge transfer across domains. Similarly, research on emotion sensing and text preprocessing techniques has demonstrated that semantic context and feature representation plays a critical role in enhancing the performance and interpretability of emotion classification systems.

Significant advances have been made in deep learning approaches, but existing systems have several limitations such as limited emotion vocabularies, weak contextual reasoning, Responsible AI, lack of transparency in predictions, and poor interpretability of model decisions. Social media textual data is also typically short, noisy and highly unstructured, with grammatical errors, slang, offensive language, irregular punctuation and intentional misspellings to evade automated filtering systems.[5] These features make the performance of conventional classification pipelines challenging, thus making accurate emotion understanding more difficult. Previous studies on emotion detection from text have studied supervised and unsupervised learning strategies by researchers and machine learning classifiers such as Naïve Bayes have been used for the detection of emotion-related information from social media texts and tweets using multiple cross-validation techniques. These traditional approaches achieved baseline classification performance, but were heavily reliant on handcrafted feature extraction and limited lexical representations. Most of the existing transformer-based emotion detection systems are also black-box models, making it difficult to identify which parts of the text contribute to the predicted emotional state.[6]  Furthermore, most of the existing approaches are limited to predefined emotion categories, and are incapable of performing open-vocabulary semantic reasoning, which is required by advanced real-world affective computing and computational emotion sensing applications [7].

Proposed Framework:

To address these limitations, we propose an explainable dual-transformer framework, EmoSphere-XAI, for fine-grained emotion reasoning from text. The proposed framework integrates the state-of-the-art ModernBERT-Large-Go-Emotions model with a custom fine tuned DistilBERT classifier for robust contextual emotion analysis. Furthermore, the framework includes a semantic embedding expansion mechanism based on Sentence Transformers and cosine similarity mapping for open vocabulary emotion understanding on a broad taxonomy of emotional concepts.

We propose a framework that uses an occlusion based Explainable Artificial Intelligence (XAI) mechanism to locate important text segments that affect emotion predictions. The system perturbs the input tokens one at a time and analyzes the change in confidence in order to give interpretable and transparent reasoning for the model’s decisions. In addition, the uncertainty quantification based on entropy is used for real-time estimation of the confidence of prediction and the trustworthiness of inference.

Significant Contributions:

Dual-transformer emotion intelligence framework for contextual emotion analysis with ModernBERT and custom DistilBERT.

Open-vocabulary emotion reasoning with semantic embedding augmentation using dense vector similarity via SentenceTransformer.

Embed occlusion-based XAI mechanism for transparent and interpretable textual emotion prediction.

Entropy-based uncertainty analysis for robust and confidence-aware emotion inference.

Developing an interactive analysis platform for real-time visualizations and explainable NLP based emotion analysis applications.s.

Sustainability and SDG Alignment

The proposed EmoSphere-XAI framework contributes to multiple UN Sustainable Development Goals (UN SDGs). The integration of explainable and emotionally intelligent AI systems supports SDG 3 (Good Health and Well-being) by offering mental health monitoring and emotionally adaptive support systems. This framework also supports SDG 4 (Quality Education) through emotionally aware intelligent tutoring and learning analytics systems . In addition, the lightweight DistilBERT architecture contributes to SDG 9 (Industry, Innovation and Infrastructure) by enabling efficient and scalable AI deployment with reduced computational needs. The explainability and transparency mechanisms embedded in the framework are also aligned with responsible and ethical AI principles on sustainable technological development.

Paper Organization

The rest of this paper is organized as follows. The proposed framework is presented in section II. The related work and existing research on textual emotion detection, transformer-based NLP, and explainable AI systems are presented in Section III. The section IV  discusses the major research gaps and challenges in present emotion recognition frameworks.

Section V describes the proposed methodology and architecture of Emotion Intelligence Framework, including the transformer-based classification, semantic emotion expansion and explainability mechanisms. Section VI describes the implementation details and experimental setup. Section VII presents the performance evaluation and analytical results of the proposed system. Finally, Section VIII concludes the research paper and allocates future directions for research and applications of intelligent emotion analysis systems.

Fig. 1. Sustainable and Explainable Dual-Transformer Emotion Intelligence Framework Architecture.

RELATED WORK

The identification of human emotions from written language has become an important area of research in Natural language processing, affective computing and intelligent text analysis. Initial studies in this domain heavily depended on handcrafted linguistic rules, pre-constructed emotional lexicons, and conventional classification techniques such as Support vector machine and Naïve Bayes models to recognize emotional patterns in textual data. These methods worked reasonably well in limited scenarios, but often failed to capture finer contextual cues, subtle sentiment variations and emotional cues that are implicitly encoded in natural language.

With the recent progress of deep learning and transformer-based architectures, great success has been achieved in textual emotion recognition. Research works such as Text-Based Emotion Recognition Using Deep Learning Approach [8] and Computational Approaches for Emotion Detection in Text demonstrated the effectiveness of deep neural networks and transformer models such as BERT, RoBERTa and DistilBERT for contextual emotion analysis. Likewise, studies such as Towards Text-based Emotion Detection: A Survey and Possible Improvements [9] have shown the increasing importance of context awareness and semantic feature extraction in modern emotion detection systems. Recent research also suggests the growing use of deep learning and transformer-based architectures to enhance emotion recognition in affective computing applications.

RESEARCH GAPS

Deep learning approaches have achieved progress but there still exist some limitations in current emotion detection systems. Emotion can be expressed through facial expressions, gestures, speech and written text, where text-based detection of emotion is a complex content-classification problem involving Natural Language Processing and Machine Learning techniques[10]. Most of existing emotion detection models are black-box architectures without explainability, which makes it difficult to interpret the prediction reasoning and understand the contribution of textual features towards emotional inference. In addition, many systems are restricted to fixed categorical emotion classes and cannot support open-vocabulary emotion understanding required for real-world intelligent applications.

Recent survey studies on textual emotion recognition have extensively reviewed state-of-the-art systems, datasets, methodologies, and computational approaches while emphasizing the limitations and future research directions in this rapidly evolving field [11]. Existing frameworks also lack uncertainty quantification and analytical visualization in real-time for reliable inference of emotion. To overcome these limitations, the proposed framework combines the latest transformer architectures, semantic embedding expansion, occlusion-based Explainable AI (XAI), and entropy-driven uncertainty analysis into a single integrated intelligent emotion analysis framework.

PROPOSED METHODOLOGY

The proposed framework provides an advanced explainable emotion intelligence architecture for fine-grained emotion detection from textual data based on transformer deep learning and semantic reasoning techniques. The system combines state-of-an-art approaches for contextual emotion classification, semantic embedding expansion, explainable AI mechanisms and uncertainty quantification in one analytical pipeline. The proposed methodology aims to improve contextual understanding, interpretability and real-time emotion analysis performance. Recent work in Natural language processing has shown that Transformers based architectures such as BERT, Transformer-XL, XLM and GPT are effective for complex language understanding tasks [12]. Among them, BERT-based architectures have attracted much attention in text-based emotion detection due to its powerful contextual learning ability and superior performance in semantic representation.

A. System Architecture

The proposed architecture is a modular multi-stage processing pipeline for emotion detection and semantic analysis. The textual data is passed through the transformer based emotion classification modules. The framework uses two different backbone models: a high-accuracy ModernBERT-based classifier trained on the GoEmotions dataset, and a lightweight custom fine-tuned DistilBERT classifier optimized for fast inference. The proposed approach is inspired by deep learning assisted semantic text analysis techniques, where Natural Language Processing and semantic embeddings[13] are used to improve the contextual emotion understanding and semantic reasoning.

The predicted emotional representations are further processed by a semantic similarity engine using SentenceTransformers for open-vocabulary expansion of more than 150 emotional concepts. Semantic vector representations and word embeddings play an important role to improve learning-based NLP systems by encoding semantic and syntactic relations among textual data.[13]. The framework also includes an Explainable Artificial Intelligence (XAI) module based on occlusion perturbation to identify the token-level emotional importance in the input text. Finally, an entropy-based uncertainty analysis is performed to estimate prediction confidence and analytical reliability.

B. ModernBERT-Based Emotion Classification Module

The main emotion classification backbone is based on the cirimus/modernbert-large-go-emotions transformer model, which is based on ModernBERT architecture and trained on the GoEmotions dataset. The model performs a context-aware emotion classification among many fine-grained emotional categories with the help of deep bidirectional attention mechanisms. ModernBERT shows superior performance compared to traditional transformer architectures in terms of contextual representation learning, semantic understanding and classification accuracy of complex emotional expressions. Previous work has confirmed that deep learning architectures outperform traditional machine learning techniques such as lexicon-based approaches, Naïve Bayes (NB), Support Vector Machine (SVM), and Neural Network (NN)[14] models for multi-class emotion classification tasks and human-computer interaction systems. The model takes the textual input and outputs probability distributions over several emotional classes.

The output of the classification is denoted as:

𝑃 (𝐸𝑖 ∣ 𝑇)                                                                        Eq. 1

where T represents the input text and Ei donates the predicted emotional category. Earlier research has also highlighted that fully automated emotion recognition continues to remain an open research challenge, requiring more reliable contextual reasoning and explainability mechanisms for dependable intelligent systems.

C. Custom DistilBERT Emotion Module

The framework implements a custom fine-tuned DistilBERT model trained on the dair-ai/emotion dataset for emotion analysis in a lightweight and computationally efficient manner. It breaks down emotions into 6 categories – joy, sadness, anger, fear, love, and surprise. The base architecture is DistilBERT which is fine-tuned using head-only fine-tuning where encoder layers freeze and only the classification head is trained for the target task. This approach greatly reduces the computational cost and retains a reliable inference performance suitable for real-time applications.

Recent studies on the limitations of existing emotion detection methods have highlighted the importance of semantic interpretation, contextual understanding and keyword-level analysis for improving real-world human–computer interaction systems.[15]  Driven by these findings, the proposed framework integrates lightweight transformer inference, semantic embedding approaches, and explainability-driven reasoning to allow for improved contextual emotion recognition and enhanced analytical transparency.

We chose DistilBERT because of its lightweight architecture, which was specifically designed to reduce the computational cost, GPU memory consumption, and inference time compared to larger transformer-based models. Moreover, training the classification head only reduces training overhead and energy consumption significantly, thus improving the efficiency of the framework for sustainable AI deployment in real-time and resource-constrained environment.

D. Semantic Emotion Expansion

The proposed framework extends beyond fixed categorical prediction by incorporating semantic emotion reasoning with SentenceTransformers. The model generates dense vector embeddings for the input text and for a taxonomy of more than 150 emotional concepts. Prior work using attention-based CNNs [16] and embedding-based representations has shown that attending to emotionally salient words leads to significant improvement in contextual emotion interpretation and classification performance.

We represent the semantic similarity calculation as follows:

cos⁡(θ)=A⋅B∥A∥∥B∥                                                        Eq. 2

where A and B are dense embedding vectors. This mechanism allows for open-vocabulary emotion understanding for applications such as social media analytics, human-computer interaction, and intelligent affective computing.

E. Mechanism of Explainable AI (XAI)

The proposed system also includes an occlusion-based Explainable Artificial Intelligence (XAI) module to improve interpretability and transparency. The framework sequentially removes words from the input text one by one and measures the change of prediction confidence for the target emotion class.[17]

The token attribution score is circumscribed as:

A(wi)=P(C∣T)-P(C∣T∖{wi})                           Eq. 3

where A(wi) is the attribution score of word wi. Words with high attribution scores, which show a strong contribution to the emotional prediction, are highlighted in the visualization module to improve analytical interpretation.

E. Entropy-Based Uncertainty Analysis

The framework estimates the prediction confidence and analytical uncertainty by calculating the information entropy over the predicted probability distribution. The more uncertain the predictions, the higher the entropy value, and the more confident the emotion classifications, the lower the entropy value. This approach is consistent with previous works that highlight the need to understand the uncertainty and limitations of emotion detection systems. Recent works have reviewed the state-of-the-art techniques, emotional models, and datasets and pointed out the existing gaps and future directions to enhance this rapidly evolving field.[18]

Theentropy calculation is defined as:

H(X)=-i=1np(xi)log⁡p(xi)                                  Eq. 4where p(xi)represents the predicted probability for emotion class xi.

This uncertainty estimation mechanism improves analytical reliability and supports more transparent emotion inference in intelligent NLP systems.

Author

M. Chunling

et al. [1]s

S. Aman &

S. Szpakowicz [2]

BERT-

Devlin et al. [3]

RoBERTa-

Liu et al. [4]

M. Saxena et al.

[Proposed]

Task Description

Emotion display in chat system using animated avatar

Annotation of Emotion by category, intensity and phrase in blog text

Contextual emotion classification via bidirectional transformer pre-training

Optimised transformer emotion recognition with dynamic masking

Real-time explainable emotion detection via dual-transformer and open-vocabulary semantic expansion

Emotion Model

Ekman (6 basic emotions)

Ekman (6 basic emotions)

Ekman; fine-tuned on GoEmotions 28-class

Ekman; multi-label taxonomy

ModernBERT-Large (28-class) + DistilBERT (6-class) + Semantic Taxonomy (150+ concepts)

Detection Approach

Keyword-based lexicon matching

Hybrid (lexicon + statistical)

Transformer fine-tuning (bidirectional)

Fine-tuned transformer (robustly optimised)

Dual-transformer fusion with semantic embedding and occlusion-based XAI

Dataset

Chat logs (proprietary)

Annotated blog corpus (LiveJournal)

GoEmotions; ISEAR

ISEAR; GoEmotions

GoEmotions + dair-ai/emotion + 150+ concept semantic taxonomy

Features

WordNet-Affect, WordNet 1.6, OMCS knowledge base

WordNet-Affect, General Inquirer, intensity annotation

Bidirectional embeddings, multi-head attention, Word Piece tokenisation

Dynamic masking, larger pre-training corpus, dense embeddings

ModernBERT, DistilBERT, SentenceTransformer (all-MiniLM-L6-v2), cosine similarity, occlusion XAI, Shannon entropy

Explainability

None

None

None — black-box model

None — black-box model

Occlusion-based XAI: token attribution + word-level heatmap + entropy confidence. Only framework with full prediction transparency

Granularity

Sentence-level

Sentence-level

Sentence-level

Sentence-level

Word, Sentence, and Semantic-level

TABLE I. Comparative Analysis of Emotion Detection Approaches

IMPLEMENTATION AND EXPERIMENTAL SETUP

The proposed Emotion Intelligence Framework was implemented using deep learning and Natural Language Processing libraries derived from Python. The system combines transformer-based emotion classification, semantic embedding analysis, explainable AI mechanisms, and real-time visualization into a unified analytical pipeline. The framework was developed with PyTorch, HuggingFace Transformers, SentenceTransformers, Plotly and a bespoke UI layer. For fine-grained contextual emotion recognition the main classification backbone uses the cirimus/modernbert-large-go-emotions model. Furthermore, a lightweight fine-tuned DistilBERT model was implemented for six-class emotion classification, namely joy, sadness, anger, fear, love, and surprise. Semantic emotion expansion was implemented using the all-MiniLM-L6-v2 SentenceTransformer model for open-vocabulary emotion reasoning.

Recent studies suggest that the quality of the dataset has a strong effect on the performance and generalizability of the emotion detection systems. Experimental results on transformer-based models such as BERT and BiLSTM indicate that variations in dataset quality metrics can induce statistically significant variation in model performance, with significant effects on prediction accuracy. Moreover, BERT and other pre-trained transformer models tend to be more robust and reliable than models trained from scratch. This highlights the importance of good quality datasets for affective computing and emotion recognition research studies [19].

To promote interpretability and analytical transparency, the framework includes an occlusion-based Explainable AI module to locate emotionally relevant tokens in the input text. This is in line with sentiment-and-semantics-driven emotion recognition approaches that have been consistently shown to outperform lexicon-based approaches and to be competitive in accuracy with supervised classifiers, especially in implicit emotion understanding from written language [20][21]. In addition, the system employs entropy-based uncertainty analysis to assess the confidence and reliability of predictions, where higher entropy indicates uncertain outputs and lower entropy indicates more confident classifications, this enhancing the robustness and trustworthiness of the framework.

Fig. 2. System Architecture of the Proposed Emotion Detection and Semantic Reasoning Framework.

A. Software and Training Configuration

The experiment is configured to run on GPU and CPU execution with CUDA acceleration if available. The custom DistilBERT model was trained with head-only fine-tuning solely to achieve computationally efficient inference performance.

Parameter

Value

Programming Language

Python 3.11

Framework

PyTorch

Learning Rate

1e-3

Backbone Model

DistilBERT

Semantic Model

all-MiniLM-L6-v2

Batch Size

64

TABLE II. Experimental Configuration and Training Parameters

B. Metrics for Evaluation

The proposed framework was evaluated using standard Natural language processing(NLP) metrics such as Accuracy, Precision, Recall and F1-Score. Explainability performance at token-level occlusion attribution and contextual understanding and prediction confidence at semantic similarity and entropy analysis were evaluated. The importance of contextual emotion analysis in intelligent learning environments was emphasized by Bustos-López et al. [22], highlighting the role of advanced NLP techniques in understanding emotional patterns from textual interactions.

RESULTS AND PERFORMANCE ANALYSIS

The proposed Emotion Intelligence Framework showed an effective performance in contextual emotion detection, semantic reasoning and explainable emotion analysis. The ModernBERT based classifier provided high quality contextual understanding for finer emotional categories while the custom DistilBERT model offered an efficient lightweight inference with reduced computational complexity. The use of transformer-based and deep learning approaches in an interdisciplinary way to enhance the contextual emotion recognition from textual data is a growing trend, according to Zad et al. [23].

The framework is able to successfully identify complex emotional expressions and semantic emotional relationships using transformer based contextual embeddings and SentenceTransformers semantic similarity mapping. The system is further extended via a semantic expansion module allowing the analysis of more than 150 emotional concepts beyond fixed categorical classification.

Fig. 3. Occlusion-Based XAI Token Attribution Heatmap for Emotion Prediction.

A. Performance Evaluation

The proposed framework was evaluated on standard Natural Language Processing performance measures like Accuracy, Precision, Recall and F1-Score. The experimental results have shown better contextual understanding and semantic interpretation as compared to traditional machine learning approaches. Kaur and Saini [24] also reported that keyword- and lexicon-based approaches are often fail to capture contextual and informal language patterns, limiting their effectiveness in real-world textual emotion analysis applications.

B. Explainability and Visualization Results

The XAI module generated token-level attribution maps that showed how much each individual word contributed to emotion prediction. Multi-dimensional radar visualizations were effective in visualizing the intensity distribution among different emotional categories. The proposed dashboard also provides real-time emotion prediction, semantic proximity analysis, entropy-based uncertainty estimation, interactive analytical visualization, and an accuracy of 86.1%. Also, Krommyda et al. [25] have shown that efficient annotation and visualization techniques improve the reliability and interpretability of emotion recognition systems for short social media texts.

Fig. 4. Radar-Based Multidimensional Emotional Intensity Visualization.

Fig. 5. Overall Performance of the Proposed Emotion Detection Model.

Fig. 6. Confusion Matrix displaying Classification Performance Across Emotion Categories.

C. Comparative Analysis

The proposed framework achieves better interpretability, semantic flexibility, and analytical transparency over classical machine learning and simple transformer-based methods. The combination of semantic embedding expansion and explainable AI mechanisms increases the reliability and practicality of textual emotion detection systems.

D. Computational Sustainability

The lightweight backbone based on DistilBERT offers faster inference and less consumption of computational resources compared to fully fine-tuned large transformer architectures. The reduced training overhead and efficient semantic reasoning improve the practical deployment of the framework in sustainable edge-AI and real-time intelligent systems.

CONCLUSION AND FUTURE WORK

Detection of emotion from textual information has become a fast-growing research area in the fields of Natural Language Processing (NLP), Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), affective computing, and sustainable human-centric AI systems. The rapid development of social media platforms, online communication systems, digital forums, and conversational agents has increased the importance of understanding the emotions in textual content for intelligent human-computer interaction, sentiment understanding, mental health analysis, recommendation systems, CNN, Bi-GRU [1], and context-aware AI applications. Traditional sentiment analysis categorizes the sentiment of a text as positive, negative or neutral, whereas emotion detection aims to recognize fine-grained emotional states such as joy, anger, sadness, fear, surprise, admiration, disappointment, and contextual affective expressions. Moreover, sustainable AI-driven emotion analysis systems can promote digital well-being, emotionally adaptive educational technologies, responsible social media monitoring, and transparent intelligent systems that support ethical and socially beneficial computing environments.

Previous computational methods for emotion detection were mostly lexicon-based methods, rule-based systems, handcrafted linguistic features, and traditional machine learning algorithms such as Support Vector Machines (SVM), Naive Bayes, Deep Learning Neural Networks [2], and Decision Trees. These approaches showed moderate classification performance but did not capture the semantic context, long-range linguistic dependencies, implicit emotional cues, sarcasm, and subtle emotional reasoning in the natural language text. Recent advances in deep learning and transformer-based architectures have greatly enhanced the ability of NLP systems by enabling contextual representation learning through attention mechanisms and large-scale pre-training strategies. Furthermore, with the increasing importance of sustainable and responsible AI, demand for computationally efficient, explainable and energy-aware transformer frameworks that can provide transparent emotion inference while enabling scalable real-time deployment in intelligent systems is also increasing.

References

Bharti, S. K., Varadhaganapathy, S., Gupta, R. K., Shukla, P. K., Bouye, M., Hingaa, S. K., & Mahmoud, A. (2022). Text‐Based Emotion Recognition Using Deep Learning Approach. Computational Intelligence and Neuroscience, 2022(1), 2645381.J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd ed., vol. 2. Oxford: Clarendon, 1892, pp.68–73.

Machová, K., Szabóova, M., Paralič, J., & Mičko, J. (2023). Detection of emotion by text analysis using machine learning. Frontiers in Psychology, 14, 1190326.

Kratzwald, B., Ilić, S., Kraus, M., Feuerriegel, S., & Prendinger, H. (2018). Deep learning for affective computing: Text-based emotion recognition in decision support. Decision support systems, 115, 24-35.R. Nicole, “Title of paper with only first word capitalized,” J. Name Stand. Abbrev., in press.

Machová, K., Mach, M., & Adamišín, K. (2022). Machine learning and lexicon approach to texts processing in the detection of degrees of toxicity in online discussions. Sensors, 22(17), 6468.

Maslej-Krešňáková, V., Sarnovský, M., Butka, P., & Machová, K. (2020). Comparison of deep learning models and various text pre-processing techniques for the toxic comments classification. Applied Sciences, 10(23), 8631.

Chenna, A., Srinivas, B., & Nagaraju, S. (2021). Emotion And Sentiment Analysis From Twitter Text. Turkish Journal of Computer and Mathematics Education, 12(12), 4614-4620.

Wang, Z., Ho, S. B., & Cambria, E. (2020). A review of emotion sensing: categorization models and algorithms. Multimedia Tools and Applications, 79(47), 35553-35582.

Patel, P., Patel, D., & Bera, M. (2023). Emotion detection in text: A deep learning approach for sentiment analysis. International Journal of Novel Research and Development, 8(10), b506-b516.

Kusal, S., Patil, S., Choudrie, J., Kotecha, K., Vora, D., & Pappas, I. (2023). A systematic review of applications of natural language processing and future challenges with special emphasis in text-based emotion detection. Artificial Intelligence Review, 56(12), 15129-15215.

Shivhare, S. N., & Khethawat, S. (2012). Emotion detection from text. arXiv preprint arXiv:1205.4944

Al Maruf, A., Khanam, F., Haque, M. M., Jiyad, Z. M., Mridha, M. F., & Aung, Z. (2024). Challenges and opportunities of text-based emotion detection: A survey. IEEE access, 12, 18416-18450.

Acheampong, F. A., Nunoo-Mensah, H., & Chen, W. (2021). Transformer models for text-based emotion detection: a review of BERT-based approaches. Artificial Intelligence Review, 54(8), 5789-5829.

Guo, J. (2022). Deep learning approach to text analysis for human emotion detection from big data. Journal of Intelligent Systems, 31(1), 113-126.

Machová, K., Szabóova, M., Paralič, J., & Mičko, J. (2023). Detection of emotion by text analysis using machine learning. Frontiers in Psychology, 14, 1190326.

Kao, E. C. C., Liu, C. C., Yang, T. H., Hsieh, C. T., & Soo, V. W. (2009, April). Towards text-based emotion detection a survey and possible improvements. In 2009 International conference on information management and engineering (pp. 70-74). IEEE.

Shrivastava, K., Kumar, S., & Jain, D. K. (2019). An effective approach for emotion detection in multimedia text data using sequence based convolutional neural network. Multimedia tools and applications, 78(20), 29607-29639.

Murthy, A. R., & Anil Kumar, K. M. (2021, March). A review of different approaches for detecting emotion from text. In IOP Conference Series: Materials Science and Engineering (Vol. 1110, No. 1, p. 012009). IOP Publishing.

Al-Saqqa, S., Abdel-Nabi, H., & Awajan, A. (2018, July). A survey of textual emotion detection. In 2018 8th international conference on computer science and information technology (CSIT) (pp. 136-142). IEEE.

de León Languré, A., & Zareei, M. (2024). Improving text emotion detection through comprehensive dataset quality analysis. IEEE Access, 12, 166512-166536.

Gupta, U., Chatterjee, A., Srikanth, R., & Agrawal, P. (2017). A sentiment-and-semantics-based approach for emotion detection in textual conversations. arXiv preprint arXiv:1707.06996

Udochukwu, O., & He, Y. (2015, June). A rule-based approach to implicit emotion detection in text. In International Conference on Applications of Natural Language to Information Systems (pp. 197-203). Cham: Springer International Publishing.

Bustos-López, M., Cruz-Ramírez, N., Guerra-Hernández, A., Sánchez-Morales, L. N., & Alor-Hernández, G. (2021). Emotion detection from text in learning environments: a review. New Perspectives on Enterprise Decision-Making Applying Artificial Intelligence Techniques, 483-508.

Zad, S., Heidari, M., James Jr, H., & Uzuner, O. (2021, May). Emotion detection of textual data: An interdisciplinary survey. In 2021 IEEE World AI IoT Congress (AIIoT) (pp. 0255-0261). IEEE.

Kaur, J., & Saini, J. R. (2014). Emotion detection and sentiment analysis in text corpus: a differential study with informal and formal writing styles. International Journal of Computer Application, ISSN, 0975-8887.

Krommyda, M., Rigos, A., Bouklas, K., & Amditis, A. (2021, March). An experimental analysis of data annotation methodologies for emotion detection in short text posted on social media. In Informatics (Vol. 8, No. 1, p. 19). MDPI.