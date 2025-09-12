-- db/schema.
CREATE TABLE predictions (
  id INT IDENTITY(1,1) PRIMARY KEY,
  ts DATETIME NOT NULL DEFAULT GETUTCDATE(),
  source NVARCHAR(50),
  file_name NVARCHAR(255),
  input_text NVARCHAR(MAX),
  cleaned_text NVARCHAR(MAX),
  model_type NVARCHAR(50),
  label NVARCHAR(50),
  proba FLOAT,
  detected_type NVARCHAR(50),
  src_lang NVARCHAR(2),
  translated BIT
);
-- ...existing code...