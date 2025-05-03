-- 增加model_trainings表的log列大小
ALTER TABLE model_trainings MODIFY COLUMN log TEXT;

-- 如果需要，也可以增加model_evaluations表的log列大小
-- ALTER TABLE model_evaluations MODIFY COLUMN log TEXT; 