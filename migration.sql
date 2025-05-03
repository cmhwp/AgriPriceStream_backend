-- 添加缺少的字段到model_trainings表
ALTER TABLE model_trainings ADD COLUMN vegetable_id INT NULL;
ALTER TABLE model_trainings ADD COLUMN history_days INT DEFAULT 30;
ALTER TABLE model_trainings ADD COLUMN prediction_days INT DEFAULT 7;
ALTER TABLE model_trainings ADD COLUMN smoothing TINYINT(1) DEFAULT 1;
ALTER TABLE model_trainings ADD COLUMN seasonality TINYINT(1) DEFAULT 1;
ALTER TABLE model_trainings ADD COLUMN sequence_length INT DEFAULT 7;

-- 可选: 添加外键约束
ALTER TABLE model_trainings ADD CONSTRAINT fk_vegetable_id FOREIGN KEY (vegetable_id) REFERENCES vegetables(id);

-- 创建model_evaluations表 (如果不存在)
CREATE TABLE IF NOT EXISTS model_evaluations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  model_id INT NOT NULL,
  algorithm VARCHAR(30) NOT NULL,
  mean_absolute_error FLOAT NOT NULL,
  mean_squared_error FLOAT NOT NULL,
  r_squared FLOAT NOT NULL,
  prediction_accuracy FLOAT NOT NULL,
  evaluation_date DATETIME NOT NULL,
  vegetable_id INT NOT NULL,
  CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES model_trainings(id),
  CONSTRAINT fk_evaluation_vegetable_id FOREIGN KEY (vegetable_id) REFERENCES vegetables(id)
); 