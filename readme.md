The dataset&nbsp;of the paper titled &quot;Automated Comment Update: How Far are We?&quot;.

This is the online repository of the paper &quot;Automated Comment Update: How Far are We?&quot; appeared in ICPC'2021. We release the source code of **H<sub>EB</sub>C<sub>UP</sub>**, the dataset used in our evaluation, as well as the experiment results.

*   Dataset: The cleaned&nbsp;dataset from Liu&#39;s ASE20 paper (i.e.,&nbsp;[Automating Just-In-Time Comment Updating](https://conf.researchr.org/details/ase-2020/ase-2020-papers/45/Automating-Just-In-Time-Comment-Updating)). We removed the three types of noisy data in the dataset as described in our paper.

*   Results:
       * Baseline_CUP.json: The readable&nbsp;result files of **CUP**.
       * HebCup_all.json: The readable&nbsp;result files of **H<sub>EB</sub>C<sub>UP.</sub>**
       * HebCup_correct.json<sub>:&nbsp;</sub>The correctly updated items&nbsp;of **H<sub>EB</sub>C<sub>UP.</sub>**
*   Source: The source code&nbsp;for running **H<sub>EB</sub>C<sub>UP</sub>**.&nbsp;

Run **H<sub>EB</sub>C<sub>UP</sub>**<sub>:</sub>


`python3 HebCup.py --dataPath path2dataset`

By default, we store the dataset in&nbsp;`./dataset`
