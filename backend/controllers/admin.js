const bcrypt = require("bcrypt");
const { getAdminByUsername } = require("../data/adminData");
const { signToken } = require("../middleware/auth");

const loginAdmin = async (req, res, next) => {
  try {
    const { nickname, contrasena } = req.body;
    const admin = await getAdminByUsername(nickname);

    if (!admin) return res.status(401).json({ error: "Credenciales inválidas" });

    const match = await bcrypt.compare(contrasena, admin.password_hash);
    if (!match) return res.status(401).json({ error: "Credenciales inválidas" });

    const token = signToken({ id: admin.id, nickname: admin.nickname, role: "admin" });
    res.status(200).json({
      token,
      user: { id: admin.id, nickname: admin.nickname, role: "admin" },
    });
  } catch (error) {
    next(error);
  }
};

module.exports = { loginAdmin };
