const { Pool } = require("pg");

const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  ssl: process.env.DB_SSL === "true" ? { rejectUnauthorized: false } : false,
});

pool
  .connect()
  .then((client) => {
    console.log("✅ Conexión a PostgreSQL establecida");
    client.release();
  })
  .catch((err) => {
    console.error("❌ Error de conexión a PostgreSQL:", err.message);
  });

module.exports = pool;
