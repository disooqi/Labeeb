================================================================
                    MADAR SHARED TASK 2019
            Arabic Fine-Grained Dialect Identification

   The Fourth Workshop for Arabic Natural Language Processing
                         (WANLP 2019)

                 Development Phase - Subtask 1
                         10 April 2019

=================================================================
MADAR-Shared-Task-Subtask-1
=================================================================

Description of Data:
--------------------

The provided development file corresponds to Corpus-26 development 
file in Salameh et al. (2018).

* Corpus-26 dev set (MADAR-Corpus26-dev.tsv) consists of 200
sentences in 26 versions  (MSA and 25 city dialects) =
200*(1 + 25) =  5,200 sentences.

*** A note on MADAR Corpus: Bouamor et al. (2018) describe
Corpus-5 and Corpus-25, which are the same as Corpus-6 and
Corpus-26, respectively, except for the addition of MSA data.

Submission Instructions:
-------------------------------------
* The data provided on Codalab for MADAR Shared Task Subtask-1 
contains only Arabic sentences. The task is to predict dialect 
labels for these Arabic sentences. For full training and 
development data, please register for the shared task 
https://sites.google.com/view/madar-shared-task/home or 
email madar.shared.task@gmail.com

* Each team is allowed to submit up to three (3) runs for development 
and three (3) runs for test phase. In other words, a team can test 
several methods or parameter settings and submit the three they prefer.

* Please structure your test results as follows:
  one file per submission, named <team><N>.subtask<i>.dev, where
  # <team> stands for your team name (please use only ASCII letters, digits and “-” or “_”)
  # <N> (1, 2 or 3) is the run number
  # <i> is the number of the subtask (1 or 2)

* The file content and format should be the same as the sample submission 
file provided and contain only the country class that the systems predict.

* Create an archive with the submission file (<team><N>.subtask<i>.dev.zip)

Shared Task Metrics and Restrictions:
-------------------------------------

The performance of submitted systems will be evaluated on
MADAR-Corpus26-test.tsv which will be made available during the
evaluation phase. MADAR-Corpus6-train.tsv and
MADAR-Corpus6-dev.tsv are provided to aid building the models.
Participants are welcome to use both of these files for training
purposes.

The training data from MADAR-Shared-Task-Subtask-2 is allowed.
External manually labelled data sets are *NOT* allowed.
However, the use of publicly available unlabelled data is allowed. 

IMPORTANT: Participants are NOT allowed to use
MADAR-Corpus26-dev.tsv for training purposes. Participants must
report the performance of their best system on
MADAR-Corpus26-dev.tsv in their Shared Task system description
paper.

The labels of dialects for the cities and MSA are defined in
Salameh et al. (2018): ALE, ALG, ALX, AMM, ASW, BAG, BAS, BEI,
BEN, CAI, DAM, DOH, FES, JED, JER, KHA, MOS, MUS, RAB, RIY, SAL,
SAN, SFX, TRI, TUN and MSA.


MADAR-Corpus-Lexicon-License.txt contains the license for using
this data.


=================================================================
References
=================================================================

[1] Salameh, Mohammad, Houda Bouamor, and Nizar Habash. "Fine-
Grained Arabic Dialect Identification." Proceedings of the
27th International Conference on Computational Linguistics.
Santa Fe, New Mexico, USA, 2018.
http://aclweb.org/anthology/C18-1113

[2] Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi
Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam
Khalifa, Fadhl Eryani, Alexander Erdmann and Kemal Oflazer.
"The MADAR Arabic Dialect Corpus and Lexicon."  Proceedings
of the Eleventh International Conference on Language Resources
and Evaluation (LREC 2018). Miyazaki, Japan, 2018.
http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf

================================================================
Copyright (c) 2019 Carnegie Mellon University Qatar
and New York University Abu Dhabi. All rights reserved.
================================================================
