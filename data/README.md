# Data

After data cleansing and POS tagging, the expected structure of files is:

```
[working directory]
 |-- data/
 |    |-- inspec/
 |    |    |-- test_postagged.json
 |    |    |-- test_src.txt
 |    |    |-- test_trg.txt
 |    |-- kp20k/
 |    |    |-- test_postagged.json
 |    |    |-- test_src.txt
 |    |    |-- test_trg.txt
 |    |    |-- train_postagged.json
 |    |    |-- train_src.txt
 |    |    |-- train_trg.txt
 |    |    |-- valid_postagged.json
 |    |    |-- valid_src.txt
 |    |    |-- valid_trg.txt
 |    |-- krapivin/
 |    |    |-- (same as inspec)
 |    |-- nus/
 |    |    |-- (same as inspec)
 |    |-- semeval/
 |    |    |-- (same as inspec)
```