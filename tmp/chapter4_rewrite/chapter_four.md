# CHAPTER FOUR

## 4.1 Introduction

This chapter presents the implemented personality prediction system, the assessment of its functions and the results of the eight experimental conditions. It describes the researcher interface, participant and questionnaire interaction, and the presentation of personality profiles. The experimental results are then considered in terms of continuous prediction, High/Low classification and the amount of text retained for analysis.

The experiments examine two comment-selection approaches, two augmentation settings and two prediction models. Baseline selection and Q-learning selection are each combined with original or GAN-augmented training data, using either Lasso/ElasticNet or a bidirectional Long Short-Term Memory model, referred to as BiLSTM. Each combination is presented separately before the effects of selection, augmentation and model choice are compared.

The evaluation involved 70 PANDORA authors, comprising 50 training authors, 10 validation authors and 10 test authors. All eight conditions used the same author partitions. Continuous prediction was assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination (R²) and Pearson correlation. These results provide the main basis for the discussion, while thresholded classification offers a further view of the predicted personality scores.

## 4.2 System Implementation

### 4.2.1 Researcher Interface

The researcher interface provides access to participant records, questionnaire information, prediction tools and experimental results. After signing in, the researcher can view the available records and identify participants whose questionnaire responses or prediction profiles have been stored. The dashboard brings these functions together so that the researcher can move from an overview of the study to the details of an individual participant.

The experiment interface allows the researcher to enter the required sample and training settings and initiate processing. An experiment-history page presents previous experiments, while the details page displays the results of the conditions belonging to the selected experiment. This arrangement makes it possible to inspect the eight conditions together without combining their values into a single result.

The interface also provides access to the dataset status and the progress of an experiment. The researcher can return to a completed experiment to review its results rather than repeat the entire training procedure. The separation between experiment history and individual profiles supports two levels of inspection: the performance of a condition across authors and the output associated with a particular author.

[Insert Figure 4.1: Researcher Interface. Show the researcher dashboard and access to experiment results.]

### 4.2.2 Participant and Questionnaire Interface

Participant entry is provided through the profile-analysis interface. The researcher enters the participant's social media handle and begins the analysis process. Where a questionnaire record is not already available, the interface presents the BFI-44 questions before continuing. The participant record is associated with the submitted responses, allowing the questionnaire scores and subsequent prediction to refer to the same individual.

This process is separate from researcher account registration. The account-registration page creates access to the research application, while participant entry identifies the person whose responses and posts are to be analysed. In the implemented system, participant enrolment takes place within the profile-analysis process rather than through a separate participant account page.

[Insert Figure 4.2: Participant Registration Interface. Show participant entry through the profile-analysis form, including the social media handle field.]

The questionnaire presents 44 statements with response options from one to five. The displayed scale runs from “Disagree a lot” to “Agree a lot”. The participant selects one response for each statement and proceeds through the questionnaire to submission. This format provides a consistent way of collecting responses without requiring the participant to calculate the five personality scores.

The submission procedure checks that all 44 responses are present and fall within the permitted range. If a response is missing or invalid, the questionnaire is displayed again with a message identifying the item that requires attention. Valid responses are scored to obtain Openness, Conscientiousness, Extraversion, Agreeableness and Neuroticism values, which are retained with the participant record. Questionnaire responses can also be imported through the researcher's survey-upload facility.

[Insert Figure 4.3: BFI-44 Questionnaire Interface. Show the numbered statements, five response options and submission control.]

The participant questionnaire remains an implemented application function, but it was not the source of the labels used in the eight PANDORA experiments. Those experiments used the personality scores supplied with the dataset. The questionnaire interface and the PANDORA evaluation therefore represent different parts of the application and should be understood within their respective uses.

### 4.2.3 Prediction Results Interface

The prediction-results interface displays the five estimated personality scores for an individual author. Where observed scores are available, they are shown alongside the predicted values. This enables the researcher to examine the size and direction of the difference for each trait instead of relying only on an overall performance measure.

The PANDORA profile page also presents the observed and predicted Low/High labels and indicates whether the classification agrees for each trait. The number of comments associated with the profile is displayed, and the researcher can return from the individual profile to the prediction-batch summary. The batch view provides an overview of the processed authors, while the profile view supports closer examination of a particular result.

[Insert Figure 4.4: Personality Prediction Results Interface. Show the observed scores, predicted OCEAN scores and trait-level Low/High results for an anonymised profile.]

The volunteer profile interface includes an OCEAN radar chart. Each axis represents one Big Five trait, allowing the displayed scores to be viewed as a complete personality profile. The numerical values remain necessary because the shape of the chart alone does not indicate the exact prediction error. The graph is therefore a visual aid to the score display rather than a separate measure of model performance.

[Insert Figure 4.5: Big Five Personality Profile Visualization. Show the implemented OCEAN radar chart with readable trait labels and a clear legend.]

The experimental findings presented below are based on the held-out PANDORA evaluation. An individual profile display illustrates the application's output, but does not by itself establish the performance of a model across the test sample. In the same way, the public demonstration output is not used as evidence for the eight-condition comparison.

## 4.3 System Testing

### 4.3.1 Functional Testing

Functional assessment concerns the operations required to obtain and display a personality prediction. These include account registration, participant entry, questionnaire submission, data loading, prediction execution, result storage and profile presentation. A successful operation requires both the expected response on the interface and the correct association of the resulting information with the relevant participant or experiment.

For account registration, the expected outcome is the creation of a valid researcher account followed by access through the sign-in page. Participant processing should retain the entered identity and link the questionnaire and prediction to that record. Questionnaire submission should accept complete responses within the permitted scale and reject an incomplete submission without treating it as a completed survey.

Data acquisition requires a distinction between the available sources. The Reddit-based experimental data were loaded from the prepared PANDORA datasets. The available experimental evidence therefore demonstrates loading of supplied Reddit comments, rather than successful live retrieval from Reddit. The existing handle-based collection interface is associated with X posts and should not be reported as a Reddit retrieval test.

Prediction execution should produce five scores for a valid author profile. Result storage should retain those scores with the selected condition, and the display should retrieve the same values when the profile is reopened. Personality-profile generation should also preserve the order and labels of the five traits. These requirements allow processing, storage and presentation to be checked together.

[Insert Table 4.1: Functional Test Cases and Observed Results. Include user registration, participant processing, BFI-44 submission, Reddit/PANDORA data loading, prediction execution, result storage, profile generation and result display. Record the input, expected outcome, observed outcome and status. Enter actual observed outcomes; leave unexecuted cases pending. Distinguish dataset loading from live Reddit retrieval.]

The completed experiments provide evidence that the prediction workflow produced and retained results. They do not, on their own, establish the outcome of every interface test. Formal acceptance of the registration, questionnaire and display functions therefore depends on the observed outcomes recorded for those individual cases. This prevents the completion of model training from being treated as proof that every application function has passed.

### 4.3.2 Experimental Pipeline Validation

The experimental pipeline processed author-level comments through selection, contextual representation, applicable augmentation and prediction. Both the pooled representation required by Lasso/ElasticNet and the ordered representation required by BiLSTM produced results. The eight completed conditions confirm that the two selection approaches and the two augmentation settings were applied across both prediction-model families.

The input sample contained 11,019 comments. The training, validation and test partitions contained 8,675, 681 and 1,663 comments respectively. No shared author identifiers were reported between any pair of partitions. This separation allowed model fitting to use training authors, threshold decisions to use validation authors and final evaluation to use test authors.

Baseline selection retained the full comment input, while Q-learning retained 7,598 comments. The selected representations were used to fit the relevant prediction models. In the augmented conditions, the sparse branch generated 50 additional training rows and the sequence branch generated 50 additional sequences. The synthetic observations were used for training and did not increase the number of real authors in the held-out sets.

[Insert Figure 4.6: Executed Experimental Prediction Pipeline. Show the completed stages from PANDORA input to selection, BERT representation, training-only augmentation where applicable, prediction, validation threshold selection and test evaluation. Indicate the separate training, validation and test paths.]

Continuous scores and validation-selected thresholds were obtained for the five traits. The thresholds were retained for the test evaluation rather than selected again from the test results. The pipeline therefore provided the outputs needed for both continuous prediction analysis and subsequent High/Low interpretation.

The review of the evaluation stage identified a limitation in classification reporting. Accuracy from three trait bands was retained for the sparse branch, while the main BiLSTM accuracy represented binary High/Low decisions. The continuous measures remained available under common definitions, but the combined binary classification comparison required alignment. This distinction is considered in Section 4.4.9 before the classification results are interpreted.

[Insert Table 4.2: Experimental Pipeline Validation Results. Cover input loading, author separation, comment selection, embedding generation, augmentation, five-score prediction, threshold selection and evaluation. State the observed output at each stage. Record the classification-measure alignment as requiring verification rather than assigning an unconditional pass.]

## 4.4 Experimental Results

The eight experiments are designated E1 to E8. This numbering identifies the combinations of selection, augmentation and prediction model; it does not represent eight folds of cross-validation. The results in this section are based on the same ten test authors. Overall MAE, RMSE, R² and Pearson values are averages of the corresponding measures across the five traits.

MAE and RMSE describe prediction error on the normalized score scale, with smaller values indicating less error. R² describes performance relative to the variation in the observed scores, while Pearson correlation describes the direction and strength of association between observed and predicted values. The measures are considered together because improvement in one does not necessarily imply improvement in the others.

### 4.4.1 Experiment 1: Baseline Selection without Augmentation using Lasso/ElasticNet

Experiment 1 used the available comment history and the original training observations to fit the regularised regression model. The fitted setting was ElasticNet within the Lasso/ElasticNet branch. It provided a reference condition against which the additional processing stages could be examined. Its overall MAE was 0.23076, RMSE was 0.26656, R² was −0.10792 and Pearson correlation was 0.04548.

[Insert Table 4.3: Experiment 1 Continuous Prediction Results. Present MAE, RMSE, R² and Pearson correlation for each trait and the overall mean.]

[Insert Figure 4.7: Experiment 1 Performance Chart. Present trait-level errors and association measures in separate panels.]

The average absolute error was approximately 0.231 on the normalized scale. RMSE was larger than MAE, indicating that the size of the errors was not uniform across the predictions. The negative mean R² and low mean correlation show that this condition did not provide strong overall correspondence with the observed personality scores.

The trait results were uneven. Openness had an MAE of 0.1285 and a correlation of 0.7511, whereas Neuroticism had an MAE of 0.3229 and a correlation of −0.5237. The favourable association for Openness was therefore not representative of the complete personality profile. The condition demonstrates why an overall average should be read alongside the individual trait results.

### 4.4.2 Experiment 2: Baseline Selection without Augmentation using BiLSTM

Experiment 2 applied BiLSTM to the baseline-selected comment sequences without synthetic training observations. The model produced five continuous personality estimates from each author's sequence. Its overall MAE was 0.23032, RMSE was 0.26252, R² was −0.07126 and Pearson correlation was 0.06724.

[Insert Table 4.4: Experiment 2 Continuous Prediction Results. Present the five trait results and overall means for MAE, RMSE, R² and Pearson correlation.]

[Insert Figure 4.8: Experiment 2 Performance Chart. Use the same layout and scales as the Experiment 1 chart.]

The recorded error remained substantial relative to the normalized target range. The mean R² was negative, while the correlation indicated only a weak positive association when averaged across traits. The sequence representation therefore produced measurable predictions, but did not establish strong continuous predictive performance across the complete OCEAN profile.

Openness recorded an MAE of 0.1377, an R² of 0.0312 and a correlation of 0.5632. Conscientiousness recorded an MAE of 0.2701 and an R² of −0.2566. These values show that retaining the comment sequence did not benefit every trait equally. The small positive Openness R² was accompanied by negative values for the other four traits, resulting in the negative overall mean.

### 4.4.3 Experiment 3: Baseline Selection with GAN Augmentation using Lasso/ElasticNet

Experiment 3 combined baseline comment selection with GAN-augmented training data and regularised regression. The original test authors were retained for evaluation, so the generated observations affected fitting rather than the composition of the test sample. The overall MAE was 0.22054, RMSE was 0.26416, R² was −0.07762 and Pearson correlation was 0.12584.

[Insert Table 4.5: Experiment 3 Continuous Prediction Results. Present trait-level and overall regression measures.]

[Insert Figure 4.9: Experiment 3 Performance Chart. Show the five trait errors and their R² and Pearson values.]

The mean correlation was positive but weak, and the negative mean R² indicated limited performance in reproducing the observed score variation. The error measures also showed that the augmented model continued to make appreciable differences from the supplied personality scores. The presence of additional training observations did not remove this limitation.

Agreeableness recorded an MAE of 0.2189 and a positive R² of 0.1142. Extraversion recorded an MAE of 0.1726, while Openness recorded 0.1184. Neuroticism remained difficult, with an MAE of 0.3453 and a correlation of −0.7061. The results suggest that the effect of the augmented training material varied across the target traits, rather than producing a uniform change in the personality estimates.

### 4.4.4 Experiment 4: Baseline Selection with GAN Augmentation using BiLSTM

Experiment 4 used baseline comment sequences and GAN augmentation with BiLSTM. Its overall MAE was 0.22652, RMSE was 0.26136, R² was −0.07288 and Pearson correlation was −0.01542. The small negative correlation indicates that the average association between predicted and observed scores was close to zero.

[Insert Table 4.6: Experiment 4 Continuous Prediction Results. Include the five trait results and overall averages.]

[Insert Figure 4.10: Experiment 4 Performance Chart. Retain separate error and association panels.]

RMSE exceeded MAE by 0.03484, reflecting the greater weight given to larger deviations. Although the model returned estimates on the personality-score scale, the negative mean R² showed that the squared-error performance remained limited. The near-zero mean correlation also provided little evidence of consistent agreement in the ordering of the test authors.

The trait correlations ranged from 0.4604 for Openness to −0.3918 for Neuroticism. Agreeableness had an R² of 0.0010, which was close to the reference level, while the other traits had negative R² values. These results indicate that the augmented sequence model did not consistently reproduce the differences between authors across all five personality dimensions.

### 4.4.5 Experiment 5: Q-Learning Selection without Augmentation using Lasso/ElasticNet

Experiment 5 used Q-learning to select comments before fitting the regularised regression model on the original training data. The overall MAE was 0.21934, RMSE was 0.26548, R² was −0.09810 and Pearson correlation was 0.19910. The positive mean correlation was accompanied by a negative mean R², showing that association and absolute score accuracy did not give the same picture of performance.

[Insert Table 4.7: Experiment 5 Continuous Prediction and Selection Results. Include trait and overall regression measures, 7,598 selected comments, 108.54 mean comments retained and 31.05 per cent comment reduction.]

[Insert Figure 4.11: Experiment 5 Prediction and Selection-Efficiency Chart. Separate prediction measures from the number of comments retained.]

Q-learning retained 7,598 of the 11,019 available comments across the sampled authors. The mean input decreased from 157.41 to 108.54 comments per author. The reduction represents a smaller quantity of text passed to representation construction, rather than an increase in the number of labelled observations or a direct measure of faster execution.

Openness recorded an MAE of 0.1257 and a correlation of 0.7156. Neuroticism recorded an MAE of 0.2686 and a correlation of 0.2799. The condition therefore retained useful association for some traits despite the reduced text input. Its negative R² values, however, indicate that the continuous estimates still differed materially from the observed scores.

### 4.4.6 Experiment 6: Q-Learning Selection without Augmentation using BiLSTM

Experiment 6 applied BiLSTM to Q-learning-selected comment sequences using the original training data. Its overall MAE was 0.22916, RMSE was 0.26764, R² was −0.11322 and Pearson correlation was −0.06094. The model used the same selected-comment set as the corresponding sparse condition, but preserved the sequence of representations rather than averaging them into a single author vector.

[Insert Table 4.8: Experiment 6 Continuous Prediction and Selection Results. Present the trait and overall measures together with selected-comment count, mean comments retained and percentage reduction.]

[Insert Figure 4.12: Experiment 6 Prediction and Selection-Efficiency Chart. Use the same presentation as Experiment 5.]

The reduced input contained 7,598 comments, corresponding to 108.54 comments per author on average. This retained-text measure describes the selection stage shared by the Q-learning conditions. The continuous result shows that the smaller input remained sufficient to produce a complete OCEAN estimate, but the negative mean correlation did not indicate reliable ordering of authors across the five traits.

Openness had an R² of 0.0545 and Agreeableness had an R² of 0.0057. The remaining traits had negative values, including −0.2689 for Conscientiousness and −0.2545 for Extraversion. The condition therefore showed limited success in some dimensions without extending that behaviour consistently to the complete profile.

### 4.4.7 Experiment 7: Q-Learning Selection with GAN Augmentation using Lasso/ElasticNet

Experiment 7 combined Q-learning selection, GAN augmentation and regularised regression. Its overall MAE was 0.22838, RMSE was 0.26672, R² was −0.14994 and Pearson correlation was 0.20504. The positive correlation showed some average agreement in score ordering, while the error and R² measures indicated that the predicted values remained imperfect estimates of the observed scores.

[Insert Table 4.9: Experiment 7 Continuous Prediction and Selection Results. Show trait and overall regression measures and the selected-comment statistics.]

[Insert Figure 4.13: Experiment 7 Prediction and Selection-Efficiency Chart. Display the error measures separately from comment retention.]

The comment-selection stage retained 68.95 per cent of the available input. Synthetic training rows were then introduced without altering the real validation or test authors. These two operations have different meanings: selection reduced the text represented for the real authors, while augmentation added generated observations to support fitting. They should not be combined into a single participant count.

Agreeableness recorded an R² of 0.1359 and a correlation of 0.4283. In contrast, Openness recorded an R² of −0.4928 despite a positive correlation of 0.3428. This contrast illustrates that a model can follow part of the score ordering while still placing the scores at inaccurate numerical levels. The complete set of measures is therefore needed to interpret the combined condition.

### 4.4.8 Experiment 8: Q-Learning Selection with GAN Augmentation using BiLSTM

Experiment 8 combined learned comment selection, GAN augmentation and BiLSTM prediction. It recorded an overall MAE of 0.22960, RMSE of 0.26142, R² of −0.06590 and Pearson correlation of 0.02560. The model retained the selected sequence representation while incorporating generated training sequences.

[Insert Table 4.10: Experiment 8 Continuous Prediction and Selection Results. Present trait and overall regression measures and the selected-comment statistics.]

[Insert Figure 4.14: Experiment 8 Prediction and Selection-Efficiency Chart. Use the same measures and scales as the other Q-learning conditions.]

The selection stage again reduced comment input by 31.05 per cent. The mean absolute error remained close to 0.23, while the mean correlation was close to zero. These values indicate that the complete combination of processing stages did not remove the difficulties encountered in predicting the observed personality scores.

Agreeableness had a small positive R² of 0.0177 and a correlation of 0.4009. Openness had a correlation of 0.5008, but its R² was −0.0449. Neuroticism had an MAE of 0.3149 and a negative correlation of −0.4244. The outcomes demonstrate variation within a single pipeline and show why the final interpretation must consider both the condition averages and the separate personality dimensions.

### 4.4.9 Threshold Determination and High/Low Classification Results

The continuous predictions were subsequently converted into High/Low categories. This provided an additional assessment of whether a predicted score fell on the appropriate side of a trait boundary. Classification does not replace continuous evaluation: a score may receive the correct category while still differing considerably from the observed value, and a small numerical error may change the category of an observation close to its boundary.

#### 4.4.9.1 Trait-Specific Thresholds

Two boundaries are relevant to the classification procedure. The observed High/Low label is defined from the training median for each trait. The predicted High/Low label is obtained by applying a validation-selected decision threshold to the model output. The observed-label boundary is common across the conditions, while the model decision threshold can differ because the conditions produce different score distributions.

The observed-label thresholds were 0.790 for Openness, 0.355 for Conscientiousness, 0.250 for Extraversion, 0.350 for Agreeableness and 0.460 for Neuroticism. These are sample-based boundaries used for evaluation. They do not represent fixed clinical levels or imply that the test sample contains equal numbers of High and Low authors for every trait.

Decision thresholds were selected by evaluating candidate boundaries on validation predictions. The selection score combined F1 and specificity as 2 × F1 × specificity ÷ (F1 + specificity). Both quantities were calculated from the validation classifications. A stronger score therefore required attention to identifying High observations while also correctly recognising Low observations.

For the ten-author validation sample, the candidate thresholds came from the distinct predicted values, rounded within the normalized range. The selected boundary was then retained for test evaluation. For example, E2 used decision thresholds of 0.7551, 0.4865, 0.3479, 0.3911 and 0.4667 for O, C, E, A and N respectively. These values differ from the observed-label medians because the two sets of boundaries serve different purposes.

[Insert Table 4.11: Big Five Trait Thresholds. Present the common observed-label cutoff for each trait and the validation-selected decision thresholds for E1–E8. Keep the two types of threshold clearly labelled.]

[Insert Figure 4.15: Trait Threshold Comparison. Show the decision thresholds across the eight conditions, with the observed-label boundary identified separately for each trait.]

#### 4.4.9.2 High/Low Classification Results

High/Low classification is assessed using accuracy and Macro-F1. Accuracy measures the proportion of correct decisions. Macro-F1 gives equal weight to the F1 values for the Low and High classes before averaging across the five traits. This is different from calculating F1 only for the High class, and the distinction is necessary when interpreting the recorded measures.

The available BiLSTM results provide some indication of binary classification behaviour. E2 recorded a mean accuracy of 0.60, with trait accuracies of 0.90 for Openness, 0.50 for Conscientiousness, 0.50 for Extraversion, 0.50 for Agreeableness and 0.60 for Neuroticism. The remaining BiLSTM conditions recorded mean accuracies of 0.44 for E4, 0.52 for E6 and 0.50 for E8. These figures describe agreement for individual trait decisions rather than the proportion of authors whose entire five-trait profile was correct.

The recorded High-class F1 for E2 was 0.5811. This value should not be presented as a verified binary Macro-F1. In addition, the sparse model's recorded accuracy represented three trait bands rather than the same binary task used in the BiLSTM accuracy results. A direct accuracy comparison between the model families would therefore combine different classification definitions.

[Insert Table 4.12: High/Low Classification Results Across Experimental Conditions. Complete accuracy and binary Macro-F1 after evaluating all eight conditions with the same observed labels and retained decision thresholds. Do not substitute High-class F1 or three-band accuracy for these measures.]

[Insert Figure 4.16: Accuracy and Macro-F1 Comparison. Plot the verified binary measures for E1–E8 using the same averaging procedure.]

The limited test sample also affects interpretation. Each trait was evaluated on ten authors, so one changed decision alters that trait's accuracy by ten percentage points. Consequently, a favourable value for one trait should be interpreted alongside its class distribution and the continuous errors. The combined classification verdict remains open until the common accuracy and Macro-F1 results have been verified.

## 4.5 Discussion of Results

### 4.5.1 Comparison of the Eight Experimental Conditions

The eight conditions produced overall MAE values between 0.21934 and 0.23076 and RMSE values between 0.26136 and 0.26764. The relatively narrow ranges indicate that the processing changes did not create large differences in average prediction error within this sample. Their effects were nevertheless uneven, and the ordering of conditions depended on the measure considered.

[Insert Table 4.13: Comparative Results of the Eight Experimental Conditions. Use columns for experiment, selection, augmentation, model, MAE, RMSE, R² and Pearson correlation. List E1–E8 in the order established in Section 4.4.]

[Insert Figure 4.17: Eight-Condition Comparative Performance Chart. Present MAE, RMSE, R² and Pearson in separate panels with consistent experiment labels.]

For example, E5 recorded a lower MAE than E2, but E2 recorded a lower RMSE. E7 had a positive mean correlation of 0.20504 while also having a mean R² of −0.14994. These combinations show that score ordering, average absolute error and larger squared errors describe different aspects of performance. No single measure gives a complete account of the predictions.

All eight mean R² values were negative. For an individual trait, a negative R² means that squared prediction error exceeds the error obtained by predicting the observed test mean for that trait. The negative averages therefore limit any claim that the models explained the observed variation well. They do not imply that every trait in every condition had a negative result, as the positive Openness and Agreeableness values demonstrate.

The comparisons are descriptive findings from one author sample. With 50 real training authors and ten test authors, small differences cannot be assumed to represent stable advantages across other samples. The final pipeline choice should therefore consider continuous prediction quality, verified classification performance and practical processing requirements together. The results below examine these factors without assigning an overall winner.

[Final pipeline verdict to be completed by the researcher: preferred condition __________; main supporting results __________; principal trade-off __________.]

### 4.5.2 Effect of Adaptive Comment Selection

Adaptive selection is assessed by comparing E1 with E5, E2 with E6, E3 with E7 and E4 with E8. Within each pair, the prediction model and augmentation setting remain unchanged. This comparison isolates the observed effect of replacing the full comment history with Q-learning-selected comments.

Without augmentation, the sparse model's MAE decreased by 0.01142 and its RMSE decreased by 0.00108 after selection. Its Pearson correlation increased from 0.04548 to 0.19910. The corresponding BiLSTM comparison showed a smaller MAE reduction of 0.00116, but RMSE increased by 0.00512 and correlation changed from 0.06724 to −0.06094. Selection therefore affected the two unaugmented model families differently.

Under GAN augmentation, selection increased MAE by 0.00784 for sparse regression and by 0.00308 for BiLSTM. RMSE increased by 0.00256 for the sparse model and changed only slightly, by 0.00006, for BiLSTM. These findings show that the improvement observed in the unaugmented sparse condition was not preserved when the same selection approach was combined with augmentation.

[Insert Table 4.14: Baseline versus Q-Learning Selection. Present the four matched pairs, changes in regression measures, mean comments retained and percentage reduction.]

[Insert Figure 4.18: Prediction Performance and Selection Efficiency. Show the change in prediction error alongside the reduction in retained comments.]

Across the sampled authors, Q-learning removed 3,421 comments from the representation input, corresponding to a reduction of 31.05 per cent. This directly addresses the objective of reducing unnecessary text processing before contextual representation. However, a smaller comment set did not guarantee improved prediction. Its value depended on whether the retained text was sufficient for the chosen prediction model and training setting.

The recorded feature-preparation times were approximately 3.23 seconds for baseline selection and 28.81 seconds for Q-learning. These times do not support an equivalent percentage reduction in elapsed processing time. Selection introduces its own work, while reuse of existing embeddings can affect the baseline time. The principal efficiency finding is therefore the reduction in comment input, with timing benefits requiring a controlled comparison under the same processing conditions.

### 4.5.3 Effect of GAN Augmentation

The effect of augmentation is examined through E1 versus E3, E2 versus E4, E5 versus E7 and E6 versus E8. These pairs retain the selection method and prediction model while changing whether synthetic observations are included during training. The test authors remain real observations and are unchanged within each pair.

Under baseline selection, augmentation reduced sparse MAE by 0.01022 and BiLSTM MAE by 0.00380. RMSE also decreased, by 0.00240 and 0.00116 respectively. The direction of the error changes was therefore favourable for both baseline-selection models, although the size of the change differed.

Under Q-learning selection, augmentation increased sparse MAE by 0.00904 and BiLSTM MAE by 0.00044. The sparse RMSE increased by 0.00124, while the BiLSTM RMSE decreased by 0.00622. The latter result suggests that augmentation altered the larger errors differently from the average absolute errors. It cannot be described simply as improvement or deterioration without specifying the measure.

[Insert Table 4.15: Augmented versus Non-Augmented Results. Show the four matched comparisons and changes in MAE, RMSE, R² and Pearson correlation.]

[Insert Figure 4.19: GAN Augmentation Performance Comparison. Present the matched changes, making the direction of improvement clear for each metric.]

The findings support treating augmentation as an experimental factor rather than an assured improvement. Generated observations reflect patterns learned from the available training data and do not provide new independent human evidence. A larger fitted training set may therefore alter the model without consistently improving held-out predictions, as the different matched outcomes show (Goodfellow et al., 2014).

Augmentation also introduced additional processing. The recorded training durations for the augmented sparse conditions were approximately 186.04 and 50.59 seconds. The augmented BiLSTM conditions recorded approximately 56,253.64 and 883.46 seconds. The unusually long baseline BiLSTM duration requires confirmation under repeated, controlled execution before it is treated as a typical training requirement. Nevertheless, the observation reinforces the need to consider computational cost alongside small changes in prediction error.

### 4.5.4 Comparison of Lasso/ElasticNet and BiLSTM

The model-family comparison uses E1 versus E2, E3 versus E4, E5 versus E6 and E7 versus E8. Each pair shares a selection and augmentation setting. The comparison concerns pooled author representations fitted by regularised regression and comment sequences processed by BiLSTM. This distinction reflects the different information retained by the two representations.

Under baseline selection without augmentation, BiLSTM reduced MAE by 0.00044 and RMSE by 0.00404 relative to sparse regression. With baseline selection and augmentation, sparse regression had the lower MAE by 0.00598, while BiLSTM had the lower RMSE by 0.00280. The relative result therefore depended on whether emphasis was placed on typical absolute error or greater sensitivity to large errors.

Under Q-learning without augmentation, sparse regression had lower MAE by 0.00982 and lower RMSE by 0.00216. Under Q-learning with augmentation, sparse regression retained a slightly lower MAE by 0.00122, but BiLSTM had a lower RMSE by 0.00530. No uniform advantage across all settings and measures was observed.

[Insert Table 4.16: Lasso/ElasticNet versus BiLSTM Results. Present each matched model pair and its regression measures.]

[Insert Figure 4.20: Model-Family Performance Comparison. Compare the two models within each of the four selection-and-augmentation settings.]

Regularised regression can restrict the influence of a large number of features when the number of training observations is limited (Friedman et al., 2010). The sequence model offers a different capacity by processing the ordered comment representations. The present results do not establish that this additional capacity consistently improved prediction with the available training sample.

The training-time comparison further distinguishes the two approaches. Without augmentation, sparse training took approximately 12.89 seconds under baseline selection and 13.65 seconds under Q-learning. The corresponding BiLSTM durations were approximately 591.96 and 256.48 seconds. These single-run timings indicate a greater fitting cost for the sequence branch, but should not be interpreted as fixed ratios that apply to every machine or author sample.

### 4.5.5 Interaction Between Selection, Augmentation and Prediction Model

The results show that selection and augmentation did not act independently. In sparse regression, Q-learning reduced MAE when no synthetic observations were added, but increased MAE under augmentation. The change from a reduction of 0.01142 to an increase of 0.00784 gives a difference of 0.01926 between the two selection effects. This describes how strongly the observed selection outcome changed with the augmentation setting.

For BiLSTM, Q-learning changed MAE by −0.00116 without augmentation and by +0.00308 with augmentation. The difference between these effects was 0.00424. The direction changed in both model families, but the recorded change was larger for sparse regression. This indicates that the usefulness of the selected comment set depended partly on the later training procedure.

[Insert Table 4.17: Combined Factor Comparison. Show the selection effect with and without GAN for each model, the corresponding augmentation effects and the differences between them. Define each change as the second condition minus its matched reference.]

[Insert Figure 4.21: Interaction between Selection, Augmentation and Prediction Model. Use separate model panels with MAE against selection method and separate lines for augmented and unaugmented training.]

The RMSE pattern also varied. For BiLSTM, Q-learning increased RMSE by 0.00512 without augmentation but changed it by only 0.00006 with augmentation. GAN therefore altered the relationship between selection and larger prediction errors. For sparse regression, the combination did not preserve the error reductions observed when selection or augmentation was added separately to the baseline.

These observations explain why the complete eight-condition design was necessary. Studying only the baseline and the fully combined pipeline would conceal the behaviour of the intermediate combinations. The interaction findings remain descriptive: they identify patterns in the recorded errors, without establishing a statistically significant interaction or proving the particular mechanism responsible for it.

### 4.5.6 Trait-Level Prediction Results

The overall comparisons summarise five related but distinct prediction tasks. Each personality trait has its own score distribution, and a condition can perform differently across them. Trait-level analysis therefore examines whether an apparent overall benefit is widespread or concentrated in one dimension. Table 4.18 brings together the separate trait measures for the eight conditions.

[Insert Table 4.18: Prediction Performance by Big Five Trait. Present MAE, RMSE, R² and Pearson for O, C, E, A and N across E1–E8. Use separate trait panels if necessary.]

[Insert Figure 4.22: Trait-Level Performance Comparison. Show all eight conditions for each trait using consistent scales.]

#### 4.5.6.1 Openness

Openness MAE ranged from 0.1184 in E3 to 0.1522 in E7. The baseline sparse model recorded 0.1285, while the baseline BiLSTM recorded 0.1377. These errors were smaller than the corresponding errors for several other traits, but the result should be read alongside the variation in observed Openness scores rather than treated as proof that Openness is universally easier to predict.

All eight conditions recorded positive Openness correlations, ranging from 0.3428 to 0.7511. However, positive R² was observed only in E2 and E6, at 0.0312 and 0.0545 respectively. E1 illustrates the distinction: its correlation was 0.7511, but its R² was −0.0854. The predictions followed the relative ordering of authors more successfully than they reproduced their numerical score levels.

E7 had the most negative Openness R², at −0.4928. This condition combined selection and augmentation, yet its Openness errors were not reduced by that combination. The trait findings therefore support examining numerical agreement separately from association when judging the effect of the added processing stages.

#### 4.5.6.2 Conscientiousness

Conscientiousness MAE ranged from 0.2475 to 0.2714, and every condition recorded a negative R². E3 had an MAE of 0.2475, while E6 recorded 0.2714. The generally negative R² values indicate that none of the tested combinations reproduced the observed variation adequately under the squared-error comparison for this trait.

The non-augmented sparse conditions E1 and E5 recorded identical MAE of 0.2619 and Pearson values of zero. The repeated values indicate that the change in selected comments did not translate into a measurable change in these Conscientiousness results. A recorded zero correlation, particularly where predictions vary little, should not be taken as evidence that Conscientiousness has no relationship with language in general.

The augmented sparse conditions recorded positive correlations of 0.1893 and 0.2905, but their R² values remained negative. For BiLSTM, the correlations ranged from −0.2357 to 0.0651. The conditions therefore showed weak or inconsistent association, with no broad improvement across the error and association measures together.

#### 4.5.6.3 Extraversion

Extraversion MAE ranged from 0.1726 in E3 to 0.2003 in E6. The baseline sparse condition recorded an MAE of 0.1955, while the baseline BiLSTM recorded 0.1930. All eight R² values were negative, indicating continued difficulty in reproducing the variation among the test authors even where the average absolute error was comparatively smaller.

The augmented baseline conditions produced lower MAE than their corresponding unaugmented baselines. E3 recorded 0.1726 and E4 recorded 0.1851. However, E3 had a positive correlation of 0.2751, whereas E4 had a negative correlation of −0.2212. Similar error magnitudes therefore did not imply similar agreement in the ranking of authors.

All four BiLSTM conditions recorded negative Extraversion correlations. The result suggests that sequence processing, as fitted in this sample, did not consistently order the authors in line with their observed Extraversion scores. This is a finding about the evaluated conditions and sample, rather than evidence that sequential text information cannot support Extraversion prediction.

#### 4.5.6.4 Agreeableness

Agreeableness displayed a different pattern from Conscientiousness and Extraversion. MAE ranged from 0.2189 to 0.2450, and several conditions had positive R² values. E3 recorded an MAE of 0.2189 and an R² of 0.1142, while E7 recorded an MAE of 0.2363 and an R² of 0.1359. E7 also recorded a correlation of 0.4283.

The contrast between E3 and E7 is useful because the lower MAE occurred in E3 while the lower RMSE and higher R² occurred in E7. Their RMSE values were 0.2770 and 0.2736 respectively. The selected and augmented condition therefore reduced the squared-error measure without producing the smallest average absolute error for this trait.

BiLSTM correlations changed from −0.1117 in E2 to 0.0918 in E4, 0.2140 in E6 and 0.4009 in E8. These values suggest more favourable ordering under the later combinations, although the corresponding R² values remained close to zero. The findings indicate some useful trait-specific behaviour without establishing strong overall score prediction.

#### 4.5.6.5 Neuroticism

Neuroticism MAE ranged from 0.2686 to 0.3453 and was generally larger than the errors observed for Openness and Extraversion. E5 recorded an MAE of 0.2686, while E3 recorded 0.3453. Every condition had a negative Neuroticism R², showing that the squared-error limitation persisted across selection, augmentation and model settings.

The sparse selection comparison was particularly noticeable. E1 recorded a correlation of −0.5237, while E5 recorded 0.2799. Adding GAN to the baseline sparse model produced a correlation of −0.7061 in E3. These changes show that the direction of association was sensitive to the processing combination, rather than consistently strengthened by additional stages.

Among the BiLSTM conditions, only E2 recorded a positive Neuroticism correlation, at 0.0511. Its R² of −0.0064 was close to zero, but the other sequence conditions recorded negative correlations. The continuous Neuroticism results therefore remain a substantial limitation of the estimated personality profiles.

Taken together, the trait-level findings show that the effects of selection and augmentation were concentrated differently across the five targets. The study met its objective of comparing the eight processing conditions and demonstrated a reduction in text input through adaptive selection. It did not establish uniformly strong prediction across the Big Five traits. The final assessment of a preferred pipeline must therefore consider the full profile, the common classification results and the processing trade-offs rather than rely on one favourable trait or measure.

## REFERENCES

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1–22.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems (Vol. 27, pp. 2672–2680).
