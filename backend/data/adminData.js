const pool = require("../config/db");

const getAdminByUsername = async (nickname) => {
  const { rows } = await pool.query(
    "SELECT id, nickname, password_hash FROM admin WHERE nickname = $1",
    [nickname]
  );
  return rows[0] || null;
};

module.exports = { getAdminByUsername };
