you can check the experiment fro the repo and tell us the function of tese var:
cfg = ExperimentConfig(sample_n_users=N_USERS,min_comments_per_user=5,seed=SEED,top_k=10,qlearning_train_epochs=3,val_ratio=0.2,synthetic_weight=0.35,gan_epochs=150,gan_batch_size=16,bert_max_length=256,lstm_epochs=35,lstm_batch_size=4,output_dir=str(RESULTS_DIR),embedding_cache_dir=str(CACHE_DIR),
)

we increase the size to 100.
we need to translate and know for a fact what our pipeline output gives and their meaining; what is good and what is not backed by our reference list search (is any of this true with what they have discovered)

Mounted at /content/drive
Users: 100

ExperimentRunner imported.
Configured experiment cells: ['lasso_baseline', 'lasso_qlearn', 'lasso_baseline_gan', 'lasso_qlearn_gan', 'lstm_baseline', 'lstm_qlearn', 'lstm_baseline_gan', 'lstm_qlearn_gan']
/content/personality-prediction-app/backend/ml_pipeline/services/data/pandora.py:3: SyntaxWarning: invalid escape sequence '\m'
  \backend\ml_pipeline\services\data\pandora.py
/content/personality-prediction-app/backend/ml_pipeline/cleaning/cleaner.py:2: SyntaxWarning: invalid escape sequence '\m'
  -\backend\ml_pipeline\cleaning\cleaner.py
WARNING:backend.ml_pipeline.services.data.pandora:Skipped 14 PANDORA rows with empty/missing text.
WARNING:backend.ml_pipeline.cleaning.cleaner:Profile field is empty or missing.

Prepared proxy-users: 1526
Cleaned comments: 1712623
ExperimentConfig(sample_n_users=100, min_comments_per_user=5, seed=42, top_k=10, qlearning_train_epochs=3, val_ratio=0.2, synthetic_weight=0.35, gan_latent_dim=64, gan_hidden_dim=128, gan_epochs=150, gan_batch_size=16, gan_learning_rate=0.0002, bert_max_length=256, lasso_alpha=0.001, lasso_l1_ratio=0.5, lasso_max_iter=10000, lasso_regularization='elasticnet', lstm_epochs=35, lstm_batch_size=4, lstm_hidden_dim=128, lstm_num_layers=2, lstm_dropout=0.2, lstm_learning_rate=0.001, output_dir='/content/drive/MyDrive/personality_prediction_lab/results/20_user_smoke_test', embedding_cache_dir='/content/drive/MyDrive/personality_prediction_lab/bert_cache')

=== baseline ===
{
  "lasso": {
    "per_trait": {
      "Openness": {
        "mae": 0.2639, "mse": 0.0953, "rmse": 0.3086,"r2": -0.0001, "correlation": 0.0
      },
      "Conscientiousness": {
        "mae": 0.2127,"mse": 0.0561,"rmse": 0.2369,  "r2": -0.4282,  "correlation": 0.0
      },
      "Extraversion": {
        "mae": 0.3372, "mse": 0.1361, "rmse": 0.369, "r2": -0.1094, "correlation": 0.0
      },
      "Agreeableness": {
        "mae": 0.2582,   "mse": 0.0903, "rmse": 0.3005, "r2": -0.0968, "correlation": -0.5035
      },
      "Neuroticism": {
        "mae": 0.2753, "mse": 0.0947,  "rmse": 0.3078,"r2": -0.0031,"correlation": 0.138
      }
    },
    "aggregate": {
      "mae": 0.2695,"mse": 0.0945,  "rmse": 0.3046,  "r2": -0.1275, "correlation": -0.0731
    }
  },
  "lstm": {
    "per_trait": {
      "Openness": {
        "accuracy": 0.5,  "precision": 0.5556,    "recall": 0.5278, "f1": 0.4963,   "specificity": 0.754,
        "confusion_matrix": [
          [    5,  1,    0   ], [  5, 2,     1     ],   [  2, 1,  3   ]   ],
        "labels": [   0, 1, 2   ]  },
      "Conscientiousness": {
        "accuracy": 0.6,  "precision": 0.4148,  "recall": 0.5091,  "f1": 0.4444, "specificity": 0.794,  "confusion_matrix": [
          [      0,      3,      1    ],    [      0,      4,      1    ],    [      1,      2,      8    ]  ],  "labels": [    0,    1,    2  ]
      },
      "Extraversion": {
        "accuracy": 0.6,
        "precision": 0.7051,
        "recall": 0.5519,
        "f1": 0.5983,
        "specificity": 0.7379,
        "confusion_matrix": [
          [     3,     0,     4   ],   [     0,     1,     1   ],   [     3,     0,     8   ]
        ],
        "labels": [   0,   1,   2
        ]
      },
      "Agreeableness": {
        "accuracy": 0.55,
        "precision": 0.5216,
        "recall": 0.5012,
        "f1": 0.4837,
        "specificity": 0.7641,
        "confusion_matrix": [
          [      1,      1,      3    ],    [      0,      7,      1    ],    [      1,      3,      3    ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Neuroticism": {
        "accuracy": 0.45,
        "precision": 0.4963,
        "recall": 0.4444,
        "f1": 0.4607,
        "specificity": 0.7308,
        "confusion_matrix": [
          [     3,     4,     0   ],   [     2,     2,     2   ],   [     0,     3,     4   ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      }
    },
    "aggregate": {  "accuracy": 0.54,  "precision": 0.5387,  "recall": 0.5069,  "f1": 0.4967,  "specificity": 0.7562
    }
  },
  "threshold": {
    "per_trait": {
      "Openness": {    "best_threshold": 0.3768,    "best_f1": 0.7097,    "accuracy": 0.55,    "precision": 0.55,    "recall": 1.0,    "f1": 0.7097,
        "results": [
          {  "threshold": 0.3768,  "accuracy": 0.55,  "precision": 0.55,  "recall": 1.0,  "f1_score": 0.7097,  "specificity": 0.0,  "tp": 11,  "fp": 9,  "tn": 0,  "fn": 0
          },
          {   "threshold": 0.3768,   "accuracy": 0.55,   "precision": 0.55,   "recall": 1.0,   "f1_score": 0.7097,   "specificity": 0.0,   "tp": 11,   "fp": 9,   "tn": 0,   "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          {   "threshold": 0.3768,   "accuracy": 0.55,   "precision": 0.55,   "recall": 1.0,   "f1_score": 0.7097,   "specificity": 0.0,   "tp": 11,   "fp": 9,   "tn": 0,   "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          }
        ],
        "ground_truth_cutoff": 0.29
      },
      "Conscientiousness": {"best_threshold": 0.5833,"best_f1": 0.6667,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,
        "f1": 0.6667,
        "results": [
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          }
        ],
        "ground_truth_cutoff": 0.775
      },
      "Extraversion": {"best_threshold": 0.3741,"best_f1": 0.6667,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1": 0.6667,
        "results": [
          { "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          {   "threshold": 0.3741,   "accuracy": 0.5,   "precision": 0.5,   "recall": 1.0,   "f1_score": 0.6667,   "specificity": 0.0,   "tp": 10,   "fp": 10,   "tn": 0,   "fn": 0
          },
          { "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          {
            "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          }
        ],
        "ground_truth_cutoff": 0.595
      },
      "Agreeableness": { "best_threshold": 0.3738, "best_f1": 0.5, "accuracy": 0.4, "precision": 0.4286, "recall": 0.6, "f1": 0.5,
        "results": [
          { "threshold": 0.3738, "accuracy": 0.4, "precision": 0.4286, "recall": 0.6, "f1_score": 0.5, "specificity": 0.2, "tp": 6, "fp": 8, "tn": 2, "fn": 4
          },
          { "threshold": 0.377, "accuracy": 0.3, "precision": 0.3333, "recall": 0.4, "f1_score": 0.3636, "specificity": 0.2, "tp": 4, "fp": 8, "tn": 2, "fn": 6
          },
          {
            "threshold": 0.3777, "accuracy": 0.3, "precision": 0.3, "recall": 0.3, "f1_score": 0.3, "specificity": 0.3, "tp": 3, "fp": 7, "tn": 3, "fn": 7
          },
          { "threshold": 0.3818, "accuracy": 0.3, "precision": 0.25, "recall": 0.2, "f1_score": 0.2222, "specificity": 0.4, "tp": 2, "fp": 6, "tn": 4, "fn": 8
          },
          { "threshold": 0.3925, "accuracy": 0.3, "precision": 0.1667, "recall": 0.1, "f1_score": 0.125, "specificity": 0.5, "tp": 1, "fp": 5, "tn": 5, "fn": 9
          }
        ],
        "ground_truth_cutoff": 0.385
      },
      "Neuroticism": { "best_threshold": 0.5257, "best_f1": 0.75, "accuracy": 0.7, "precision": 0.6429, "recall": 0.9, "f1": 0.75,
        "results": [
          { "threshold": 0.5257, "accuracy": 0.7, "precision": 0.6429, "recall": 0.9, "f1_score": 0.75, "specificity": 0.5, "tp": 9, "fp": 5, "tn": 5, "fn": 1
          },
          {"threshold": 0.5324,"accuracy": 0.7,"precision": 0.6667,"recall": 0.8,"f1_score": 0.7273,"specificity": 0.6,"tp": 8,"fp": 4,"tn": 6,"fn": 2
          },
          {"threshold": 0.5418,"accuracy": 0.7,"precision": 0.7,"recall": 0.7,"f1_score": 0.7,"specificity": 0.7,"tp": 7,"fp": 3,"tn": 7,"fn": 3
          },
          {
            "threshold": 0.56, "accuracy": 0.6, "precision": 0.625, "recall": 0.5, "f1_score": 0.5556, "specificity": 0.7, "tp": 5, "fp": 3, "tn": 7, "fn": 5
          },
          { "threshold": 0.5718, "accuracy": 0.55, "precision": 0.5714, "recall": 0.4, "f1_score": 

=== qlearning ===
{
  "lasso": {
    "per_trait": {
      "Openness": {"mae": 0.2639,"mse": 0.0953,"rmse": 0.3086,"r2": -0.0001,"correlation": 0.0
      },
      "Conscientiousness": { "mae": 0.2127, "mse": 0.0561, "rmse": 0.2369, "r2": -0.4282, "correlation": 0.0
      },
      "Extraversion": {  "mae": 0.3281,  "mse": 0.1348,  "rmse": 0.3671,  "r2": -0.0982,  "correlation": 0.049
      },
      "Agreeableness": { "mae": 0.2677, "mse": 0.0975, "rmse": 0.3123, "r2": -0.1846, "correlation": -0.0569
      },
      "Neuroticism": {
        "mae": 0.2789,
        "mse": 0.0959,
        "rmse": 0.3097,
        "r2": -0.0157,
        "correlation": 0.0
      }
    },
    "aggregate": {
      "mae": 0.2703,
      "mse": 0.0959,
      "rmse": 0.3069,
      "r2": -0.1454,
      "correlation": -0.0016
    }
  },
  "lstm": {
    "per_trait": {
      "Openness": {
        "accuracy": 0.45,
        "precision": 0.4879,
        "recall": 0.4583,
        "f1": 0.444,
        "specificity": 0.7302,
        "confusion_matrix": [
          [     2,     1,     3   ],   [     1,     3,     4   ],   [     1,     1,     4   ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Conscientiousness": {
        "accuracy": 0.65,
        "precision": 0.7157,
        "recall": 0.4833,
        "f1": 0.4905,
        "specificity": 0.7556,
        "confusion_matrix": [    [      1,      1,      2    ],    [      0,      1,      4    ],    [      0,      0,      11    ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Extraversion": {
        "accuracy": 0.6,
        "precision": 0.5476,
        "recall": 0.5346,
        "f1": 0.5067,
        "specificity": 0.7521,
        "confusion_matrix": [
          [     2,     1,     4   ],   [     0,     1,     1   ],   [     1,     1,     9   ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Agreeableness": {
        "accuracy": 0.45,
        "precision": 0.4515,
        "recall": 0.4369,
        "f1": 0.43,
        "specificity": 0.7154,
        "confusion_matrix": [  [    2,    3,    0  ],  [    1,    5,    2  ],  [    2,    3,    2  ]],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Neuroticism": {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.4921,
        "f1": 0.484,
        "specificity": 0.7491,
        "confusion_matrix": [   [     3,     2,     2   ],   [     1,     2,     3   ],   [     1,     1,     5   ] ],
        "labels": [
          0,
          1,
          2
        ]
      }
    },
    "aggregate": { "accuracy": 0.53, "precision": 0.5405, "recall": 0.481, "f1": 0.471, "specificity": 0.7405
    }
  },
  "threshold": {
    "per_trait": {
      "Openness": {  "best_threshold": 0.3768,  "best_f1": 0.7097,  "accuracy": 0.55,  "precision": 0.55,  "recall": 1.0,  "f1": 0.7097,
        "results": [
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          { "threshold": 0.3768,  "accuracy": 0.55,  "precision": 0.55,  "recall": 1.0,  "f1_score": 0.7097,  "specificity": 0.0,  "tp": 11,  "fp": 9,  "tn": 0,  "fn": 0
          }
        ],
        "ground_truth_cutoff": 0.29
      },
      "Conscientiousness": { "best_threshold": 0.5833, "best_f1": 0.6667, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667,
        "results": [
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          {  "threshold": 0.5833,  "accuracy": 0.5,  "precision": 0.5,  "recall": 1.0,  "f1_score": 0.6667,  "specificity": 0.0,  "tp": 10,  "fp": 10,  "tn": 0,  "fn": 0
          },
          {  "threshold": 0.5833,  "accuracy": 0.5,  "precision": 0.5,  "recall": 1.0,  "f1_score": 0.6667,  "specificity": 0.0,  "tp": 10,  "fp": 10,  "tn": 0,  "fn": 0
          },
          {  "threshold": 0.5833,  "accuracy": 0.5,  "precision": 0.5,  "recall": 1.0,  "f1_score": 0.6667,  "specificity": 0.0,  "tp": 10,  "fp": 10,  "tn": 0,  "fn": 0
          },
          {  "threshold": 0.5833,  "accuracy": 0.5,  "precision": 0.5,  "recall": 1.0,  "f1_score": 0.6667,  "specificity": 0.0,  "tp": 10,  "fp": 10,  "tn": 0,  "fn": 0
          }
        ],
        "ground_truth_cutoff": 0.775
      },
      "Extraversion": { "best_threshold": 0.3763, "best_f1": 0.6, "accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6,
        "results": [
          {   "threshold": 0.352,   "accuracy": 0.4,   "precision": 0.4286,   "recall": 0.6,   "f1_score": 0.5,   "specificity": 0.2,   "tp": 6,   "fp": 8,   "tn": 2,   "fn": 4
          },
          {
            "threshold": 0.3624,  "accuracy": 0.5,  "precision": 0.5,  "recall": 0.6,  "f1_score": 0.5455,  "specificity": 0.4,  "tp": 6,  "fp": 6,  "tn": 4,  "fn": 4
          },
          {   "threshold": 0.3763,   "accuracy": 0.6,   "precision": 0.6,   "recall": 0.6,   "f1_score": 0.6,   "specificity": 0.6,   "tp": 6,   "fp": 4,   "tn": 6,   "fn": 4
          },
          {"threshold": 0.3974,"accuracy": 0.5,"precision": 0.5,"recall": 0.4,"f1_score": 0.4444,"specificity": 0.6,"tp": 4,"fp": 4,"tn": 6,"fn": 6
          },
          {  "threshold": 0.4237,  "accuracy": 0.4,  "precision": 0.3333,  "recall": 0.2,  "f1_score": 0.25,  "specificity": 0.6,  "tp": 2,  "fp": 4,  "tn": 6,  "fn": 8
          }
        ],
        "ground_truth_cutoff": 0.595
      },
      "Agreeableness": { "best_threshold": 0.3192, "best_f1": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 0.7, "f1": 0.5833, "results": [
          { "threshold": 0.3192, "accuracy": 0.5, "precision": 0.5, "recall": 0.7, "f1_score": 0.5833, "specificity": 0.3, "tp": 7, "fp": 7, "tn": 3, "fn": 3
          },
          {"threshold": 0.3474,"accuracy": 0.5,"precision": 0.5,"recall": 0.6,"f1_score": 0.5455,"specificity": 0.4,"tp": 6,"fp": 6,"tn": 4,"fn": 4
          },
          {"threshold": 0.3697,"accuracy": 0.4,"precision": 0.4,"recall": 0.4,"f1_score": 0.4,"specificity": 0.4,"tp": 4,"fp": 6,"tn": 4,"fn": 6
          },
          { "threshold": 0.4181, "accuracy": 0.4, "precision": 0.375, "recall": 0.3, "f1_score": 0.3333, "specificity": 0.5, "tp": 3, "fp": 5, "tn": 5, "fn": 7
          },
          { "threshold": 0.4786, "accuracy": 0.4, "precision": 0.3333, "recall": 0.2, "f1_score": 0.25, "specificity": 0.6, "tp": 2, "fp": 4, "tn": 6, "fn": 8
          }
        ],
        "ground_truth_cutoff": 0.385
      },
      "Neuroticism": {"best_threshold": 0.538,"best_f1": 0.0,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,
        "f1": 0.0,
        "results": [
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          { "threshold": 0.538, "accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "specificity": 1.0,
     

=== baseline + GAN ===
{
  "lasso": {
    "per_trait": {
      "Openness": { "mae": 0.2639, "mse": 0.0953, "rmse": 0.3086, "r2": -0.0001, "correlation": 0.0
      },
      "Conscientiousness": { "mae": 0.2127, "mse": 0.0561, "rmse": 0.2369, "r2": -0.4282, "correlation": 0.0
      },
      "Extraversion": {  "mae": 0.3372,  "mse": 0.1361,  "rmse": 0.369,  "r2": -0.1094,  "correlation": 0.0
      },
      "Agreeableness": {"mae": 0.246,"mse": 0.0833,"rmse": 0.2887,"r2": -0.0119,"correlation": 0.0
      },
      "Neuroticism": {"mae": 0.2767,"mse": 0.0943,"rmse": 0.3071,"r2": 0.0014,"correlation": 0.1572
      }
    },
    "aggregate": {
      "mae": 0.2673,
      "mse": 0.093,
      "rmse": 0.3021,
      "r2": -0.1096,
      "correlation": 0.0314
    }
  },
  "lstm": {
    "per_trait": {
      "Openness": {
        "accuracy": 0.55,
        "precision": 0.5333,
        "recall": 0.5972,
        "f1": 0.5071,
        "specificity": 0.7817,
        "confusion_matrix": [
          [
            4,
            1,
            1
          ],
          [
            4,
            1,
            3
          ],
          [
            0,
            0,
            6
          ]
        ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Conscientiousness": {
        "accuracy": 0.6,
        "precision": 0.4,
        "recall": 0.4727,
        "f1": 0.4308,
        "specificity": 0.7333,
        "confusion_matrix": [
          [      0,      0,      4    ],    [      0,      3,      2    ],    [      0,      2,      9    ]  ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Extraversion": {"accuracy": 0.55,"precision": 0.1833,"recall": 0.3333,"f1": 0.2366,"specificity": 0.6667,
        "confusion_matrix": [
          [     0,     0,     7   ],   [     0,     0,     2   ],   [     0,     0,     11   ] ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Agreeableness": {
        "accuracy": 0.4,
        "precision": 0.319,
        "recall": 0.4583,
        "f1": 0.3293,
        "specificity": 0.7188,
        "confusion_matrix": [
          [     5,     0,     0   ],   [     4,     3,     1   ],   [     5,     2,     0   ] ],
        "labels": [
          0,
          1,
          2
        ]
      },
      "Neuroticism": { "accuracy": 0.55, "precision": 0.5519, "recall": 0.5476, "f1": 0.5417, "specificity": 0.7747,
        "confusion_matrix": [
          [
            3,
            1,
            3
          ],
          [2,3,1
          ],
          [0,2,5
          ]
        ],
        "labels": [   0,   1,    2
        ]
      }
    },
    "aggregate": { "accuracy": 0.53, "precision": 0.3975, "recall": 0.4818, "f1": 0.4091, "specificity": 0.735
    }
  },
  "threshold": {
    "per_trait": {
      "Openness": { "best_threshold": 0.3768, "best_f1": 0.7097, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1": 0.7097,
        "results": [
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.29
      },
      "Conscientiousness": { "best_threshold": 0.5833, "best_f1": 0.6667, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667,
        "results": [
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.5833, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.775
      },
      "Extraversion": { "best_threshold": 0.3741, "best_f1": 0.6667, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667,
        "results": [
          { "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          { "threshold": 0.3741, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1_score": 0.6667, "specificity": 0.0, "tp": 10, "fp": 10, "tn": 0, "fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.595
      },
      "Agreeableness": { "best_threshold": 0.3747, "best_f1": 0.6667, "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667,
        "results": [
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.385
      },
      "Neuroticism": {"best_threshold": 0.5311,"best_f1": 0.75,"accuracy": 0.7,"precision": 0.6429,"recall": 0.9,"f1": 0.75,
        "results": [
          {"threshold": 0.5311,"accuracy": 0.7,"precision": 0.6429,"recall": 0.9,"f1_score": 0.75,"specificity": 0.5,"tp": 9,"fp": 5,"tn": 5,"fn": 1
          },
          {"threshold": 0.5394,"accuracy": 0.7,"precision": 0.6667,"recall": 0.8,"f1_score": 0.7273,"specificity": 0.6,"tp": 8,"fp": 4,"tn": 6,"fn": 2
          },
          {"threshold": 0.5407,"accuracy": 0.7,"precision": 0.7,"recall": 0.7,"f1_score": 0.7,"specificity": 0.7,"tp": 7,"fp": 3,"tn": 7,"fn": 3
          },
          {"threshold": 0.5426,"accuracy": 0.7,"precision": 0.75,"recall": 0.6,"f1_score": 0.6667,"specificity": 0.8,"tp": 6,"fp": 2,"tn": 8,"fn": 4
          },
          { "threshold": 0.5441, "accuracy": 0.6, "precision": 0.6667, "recall": 0.4, "f1_score

=== qlearning + GAN ===
{
  "lasso": {
    "per_trait": {
      "Openness": {"mae": 0.2639,"mse": 0.0953,"rmse": 0.3086,"r2": -0.0001,"correlation": 0.0
      },
      "Conscientiousness": {"mae": 0.2127,"mse": 0.0561,"rmse": 0.2369,"r2": -0.4282,"correlation": 0.0
      },
      "Extraversion": {"mae": 0.3372,"mse": 0.1361,"rmse": 0.369,"r2": -0.1094,"correlation": 0.0
      },
      "Agreeableness": {"mae": 0.246,"mse": 0.0833,"rmse": 0.2887,"r2": -0.0119,"correlation": 0.0
      },
      "Neuroticism": {"mae": 0.2789,"mse": 0.0959,"rmse": 0.3097,"r2": -0.0157,"correlation": 0.0
      }
    },
    "aggregate": {"mae": 0.2677,"mse": 0.0933,"rmse": 0.3026,"r2": -0.1131,"correlation": 0.0
    }
  },
  "lstm": {
    "per_trait": {
      "Openness": {"accuracy": 0.25,"precision": 0.1717,"recall": 0.2778,"f1": 0.2118,"specificity": 0.6429,"confusion_matrix": [
          [2,0,4 ],  [6,0,2  ],  [3,0,3   ]   ],
        "labels": [ 0, 1, 2]
      },
      "Conscientiousness": {"accuracy": 0.45,"precision": 0.3407,"recall": 0.4182,"f1": 0.3333,"specificity": 0.7259,
        "confusion_matrix": [
          [ 0, 3, 1  ],   [ 0, 4, 1   ], [ 0, 6, 5  ]
        ],
        "labels": [    0,    1,    2  ]
      },
      "Extraversion": {"accuracy": 0.7,"precision": 0.6389,"recall": 0.6299,"f1": 0.6327,"specificity": 0.8191,
        "confusion_matrix": [
          [4,1,2 ], [ 0, 1, 1  ],  [ 2, 0, 9
          ]
        ],
        "labels": [ 0, 1, 2
        ]
      },
      "Agreeableness": {"accuracy": 0.4,"precision": 0.1333,"recall": 0.3333,"f1": 0.1905,"specificity": 0.6667,
        "confusion_matrix": [
          [ 0, 5, 0   ],   [0,8,0 ],  [0,7,0   ]
        ],
        "labels": [0,1,2  ]
      },
      "Neuroticism": {"accuracy": 0.5,"precision": 0.3958,"recall": 0.4762,"f1": 0.3847,"specificity": 0.7436,
        "confusion_matrix": [
          [3,0,4   ],
          [1,0,5
          ],
          [0,0,7
          ]
        ],
        "labels": [0,1,2
        ]
      }
    },
    "aggregate": {"accuracy": 0.46,"precision": 0.3361,"recall": 0.4271,"f1": 0.3506,"specificity": 0.7196
    }
  },
  "threshold": {
    "per_trait": {
      "Openness": {"best_threshold": 0.3768,"best_f1": 0.7097,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1": 0.7097,
        "results": [
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          },
          { "threshold": 0.3768, "accuracy": 0.55, "precision": 0.55, "recall": 1.0, "f1_score": 0.7097, "specificity": 0.0, "tp": 11, "fp": 9, "tn": 0, "fn": 0
          },
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          },
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          },
          {"threshold": 0.3768,"accuracy": 0.55,"precision": 0.55,"recall": 1.0,"f1_score": 0.7097,"specificity": 0.0,"tp": 11,"fp": 9,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.29
      },
      "Conscientiousness": {"best_threshold": 0.5833,"best_f1": 0.6667,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1": 0.6667,
        "results": [
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.5833,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.775
      },
      "Extraversion": {"best_threshold": 0.3741,"best_f1": 0.6667,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1": 0.6667,
        "results": [
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3741,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.595
      },
      "Agreeableness": {"best_threshold": 0.3747,"best_f1": 0.6667,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1": 0.6667,
        "results": [
          {
            "threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          },
          {"threshold": 0.3747,"accuracy": 0.5,"precision": 0.5,"recall": 1.0,"f1_score": 0.6667,"specificity": 0.0,"tp": 10,"fp": 10,"tn": 0,"fn": 0
          }
        ],
        "ground_truth_cutoff": 0.385
      },
      "Neuroticism": {"best_threshold": 0.538,"best_f1": 0.0,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,
        "f1": 0.0,
        "results": ["threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          {"threshold": 0.538,"accuracy": 0.5,"precision": 0.0,"recall": 0.0,"f1_score": 0.0,"specificity": 1.0,"tp": 0,"fp": 0,"tn": 10,"fn": 10
          },
          { "threshold": 0.538, "accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,


{
  "best_condition": {"condition": "lstm_baseline","label": "LSTM | baseline-select","accuracy": 0.54,"macro_f1": 0.4966800000000001
  },
  "model_means": {
    "lasso": {
      "accuracy": 0.2875, "macro_f1": 0.14947499999999997
    },
    "lstm": {
      "accuracy": 0.515, macro_f1": 0.431855
    }
  },
  "better_model": {
    "by_accuracy": "LSTM",   "by_macro_f1": "LSTM"
  },
  "qlearning_effect_mean": {
    "delta_accuracy": -0.017500000000000016,  "delta_macro_f1": -0.01525
  },
  "gan_effect_mean": {
    "delta_accuracy": -0.017500000000000016,   "delta_macro_f1": -0.05574000000000005
  },
  "notes": [
    "Best condition: lstm_baseline (LSTM | baseline-select) - tertile accuracy 0.540, macro-F1 0.497.",
    "Model comparison (mean over the 4 matched cells): Lasso accuracy 0.287 vs LSTM 0.515 -> LSTM wins on accuracy; Lasso macro-F1 0.149 vs LSTM 0.432 -> LSTM wins on macro-F1.",
    "Q-learning selection hurts on average: mean delta accuracy -0.018, mean delta macro-F1 -0.015 (vs baseline-select, over model x GAN).",
    "GAN augmentation hurts on average: mean delta accuracy -0.018, mean delta macro-F1 -0.056 (vs no-GAN, over model x selection)."
  ]
}
