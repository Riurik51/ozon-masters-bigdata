add archive projects/2/predict.py;
add archive projects/2/model.py;
add archive 2.joblib;
insert into hw2_pred select * from (select transform(*) using 'predict.py' as (id, pred) from
(select * from (select id,
nvl(if1, ''), nvl(if2, ''), nvl(if3, ''),
nvl(if4, ''), nvl(if5, ''), nvl(if6, ''),
nvl(if7, ''), nvl(if8, ''), nvl(if9, ''),
nvl(if10, ''), nvl(if11, ''), nvl(if12, ''),
nvl(if13, ''), nvl(cf1, ''), nvl(cf2, ''),
nvl(cf3, ''), nvl(cf4, ''), nvl(cf5, ''),
nvl(cf6, ''), nvl(cf7,''), nvl(cf8, ''),
nvl(cf9, ''), nvl(cf10,''), nvl(cf11, ''),
nvl(cf12, ''), nvl(cf13, ''), nvl(cf14, ''),
nvl(cf15, ''), nvl(cf16, ''), nvl(cf17, ''),
nvl(cf18, ''), nvl(cf19, ''), nvl(cf20, ''),
nvl(cf21, ''), nvl(cf22, ''), nvl(cf23, ''),
nvl(cf24, ''), nvl(cf25, ''), nvl(cf26, ''),
nvl(day_number, '') from (select * from hw2_test where nvl(if1, 0) > 20 and nvl(if1, 0) < 40) temp) temp) temp) temp;
