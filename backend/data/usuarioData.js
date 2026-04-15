const pool = require("../config/db");
const bcrypt = require("bcrypt");

const SALT_ROUNDS = 10;
const PUBLIC_FIELDS = "id, nombre, correo, nickname, creado_en";

const createUsuario = async (nombre, correo, nickname, contrasena, timestamp) => {
  const hash = await bcrypt.hash(contrasena, SALT_ROUNDS);
  const { rows } = await pool.query(
    `INSERT INTO usuarios (nombre, correo, nickname, password_hash, creado_en)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING ${PUBLIC_FIELDS}`,
    [nombre, correo, nickname, hash, timestamp]
  );
  return rows[0];
};

const getUsuarios = async () => {
  const { rows } = await pool.query(`SELECT ${PUBLIC_FIELDS} FROM usuarios ORDER BY id`);
  return rows;
};

const loginUsuarios = async (nickname, contrasena) => {
  const { rows } = await pool.query(
    "SELECT id, nombre, nickname, password_hash FROM usuarios WHERE nickname = $1",
    [nickname]
  );
  if (rows.length === 0) return null;

  const user = rows[0];
  const match = await bcrypt.compare(contrasena, user.password_hash);
  if (!match) return null;

  delete user.password_hash;
  return user;
};

const deleteUsuario = async (id) => {
  const { rows } = await pool.query(
    `DELETE FROM usuarios WHERE id = $1 RETURNING ${PUBLIC_FIELDS}`,
    [id]
  );
  return rows[0] || null;
};

const updateUsuario = async (id, { nombre, correo, nickname, contrasena }) => {
  const hash = contrasena ? await bcrypt.hash(contrasena, SALT_ROUNDS) : null;
  const { rows } = await pool.query(
    `UPDATE usuarios
     SET nombre = COALESCE($1, nombre),
         correo = COALESCE($2, correo),
         nickname = COALESCE($3, nickname),
         password_hash = COALESCE($4, password_hash)
     WHERE id = $5
     RETURNING ${PUBLIC_FIELDS}`,
    [nombre, correo, nickname, hash, id]
  );
  return rows[0] || null;
};

module.exports = {
  createUsuario,
  loginUsuarios,
  getUsuarios,
  deleteUsuario,
  updateUsuario,
};
