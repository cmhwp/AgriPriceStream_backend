-- 去除model_trainings表的vegetable_id外键约束
ALTER TABLE model_trainings DROP FOREIGN KEY fk_vegetable_id;

-- 调整model_trainings参数默认值
ALTER TABLE model_trainings MODIFY history_days INT DEFAULT 365;

-- 更新ModelEvaluation表，使vegetable_id可以为空（全局模型评估不针对特定蔬菜）
ALTER TABLE model_evaluations MODIFY vegetable_id INT NULL;
ALTER TABLE model_evaluations DROP FOREIGN KEY fk_evaluation_vegetable_id;
ALTER TABLE model_evaluations ADD CONSTRAINT fk_evaluation_vegetable_id 
    FOREIGN KEY (vegetable_id) REFERENCES vegetables(id) ON DELETE SET NULL; 