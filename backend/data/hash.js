const pool = require("../config/db");
const bcrypt = require("bcrypt");

async function encryptPasswords() {
  const { rows } = await pool.query("SELECT id, contraseña FROM usuarios");
  for (const user of rows) {
    // Solo encripta si la contraseña no parece ya un hash bcrypt
    if (!user.contraseña.startsWith("$2b$")) {
      const hashed = await bcrypt.hash(user.contraseña, 10);
      await pool.query(
        "UPDATE usuarios SET contraseña = $1 WHERE id = $2",
        [hashed, user.id]
      );
      console.log(`Contraseña actualizada para usuario ID: ${user.id}`);
    }
  }
  console.log("Todas las contraseñas han sido encriptadas.");
  process.exit();
}

encryptPasswords();