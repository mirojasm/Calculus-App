const jwt = require("jsonwebtoken");

const getSecret = () => {
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("JWT_SECRET no está configurado");
  return secret;
};

const signToken = (payload) =>
  jwt.sign(payload, getSecret(), {
    expiresIn: process.env.JWT_EXPIRES_IN || "8h",
  });

const authRequired = (req, res, next) => {
  const header = req.headers.authorization || "";
  const [scheme, token] = header.split(" ");

  if (scheme !== "Bearer" || !token) {
    return res.status(401).json({ error: "Token requerido" });
  }

  try {
    req.user = jwt.verify(token, getSecret());
    next();
  } catch (err) {
    return res.status(401).json({ error: "Token inválido o expirado" });
  }
};

const adminRequired = (req, res, next) => {
  authRequired(req, res, () => {
    if (req.user?.role !== "admin") {
      return res.status(403).json({ error: "Requiere permisos de admin" });
    }
    next();
  });
};

module.exports = { signToken, authRequired, adminRequired };
