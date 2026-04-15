-- Schema inicial (se ejecuta automáticamente al crear el volumen de Postgres).

CREATE TABLE IF NOT EXISTS usuarios (
  id SERIAL PRIMARY KEY,
  nombre TEXT NOT NULL,
  correo TEXT UNIQUE NOT NULL,
  nickname TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  creado_en TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS admin (
  id SERIAL PRIMARY KEY,
  nickname TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS acciones (
  id SERIAL PRIMARY KEY,
  usuario_id INTEGER REFERENCES usuarios(id) ON DELETE CASCADE,
  accion TEXT,
  current_exercise INTEGER,
  creado_en TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analisis (
  id SERIAL PRIMARY KEY,
  usuario_id INTEGER REFERENCES usuarios(id) ON DELETE CASCADE,
  analisis TEXT,
  puntaje NUMERIC,
  creado_en TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversaciones (
  id SERIAL PRIMARY KEY,
  usuario_id INTEGER REFERENCES usuarios(id) ON DELETE CASCADE,
  conversation_id TEXT,
  mensaje_usuario TEXT,
  respuesta_chatbot TEXT,
  current_exercise INTEGER,
  creado_en TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversaciones_usuario_ejercicio
  ON conversaciones (usuario_id, current_exercise);

CREATE TABLE IF NOT EXISTS resultados (
  id SERIAL PRIMARY KEY,
  usuario_id INTEGER REFERENCES usuarios(id) ON DELETE CASCADE,
  resultado_alumno TEXT,
  resultado_correcto TEXT,
  estado_ejercicio TEXT,
  current_exercise INTEGER,
  creado_en TIMESTAMP DEFAULT NOW()
);

-- Admin por defecto: nickname=admin, password=admin123
INSERT INTO admin (nickname, password_hash)
VALUES ('admin', '$2b$10$LIPYTHRdWSJs5Iofg3l5wuVvvLIiU4kEKtsUn8g0EwZzQ.1/d3Dn6')
ON CONFLICT (nickname) DO NOTHING;
