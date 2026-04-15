const {
  createUsuario,
  loginUsuarios,
  getUsuarios,
  deleteUsuario,
  updateUsuario,
} = require("../data/usuarioData");
const { signToken } = require("../middleware/auth");

const loginUsuario = async (req, res, next) => {
  try {
    const { nickname, contrasena } = req.body;
    const usuario = await loginUsuarios(nickname, contrasena);
    if (!usuario) return res.status(401).json({ error: "Credenciales inválidas" });

    const user = { id: usuario.id, nombre: usuario.nombre, nickname: usuario.nickname };
    const token = signToken({ ...user, role: "user" });
    res.status(200).json({ token, user });
  } catch (error) {
    next(error);
  }
};

const addUsuario = async (req, res, next) => {
  try {
    const { nombre, correo, nickname, contrasena } = req.body;
    const usuario = await createUsuario(nombre, correo, nickname, contrasena, new Date());
    res.status(201).json({
      id: usuario.id,
      nombre: usuario.nombre,
      correo: usuario.correo,
      nickname: usuario.nickname,
      creado_en: usuario.creado_en,
    });
  } catch (error) {
    next(error);
  }
};

const obtenerUsuarios = async (req, res, next) => {
  try {
    const usuarios = await getUsuarios();
    res.status(200).json(usuarios);
  } catch (error) {
    next(error);
  }
};

const eliminarUsuario = async (req, res, next) => {
  try {
    const eliminado = await deleteUsuario(req.params.id);
    if (!eliminado) return res.status(404).json({ error: "Usuario no encontrado" });
    res.status(200).json(eliminado);
  } catch (error) {
    next(error);
  }
};

const modificarUsuario = async (req, res, next) => {
  try {
    const { nombre, correo, nickname, contrasena } = req.body;
    const actualizado = await updateUsuario(req.params.id, {
      nombre,
      correo,
      nickname,
      contrasena,
    });
    if (!actualizado) return res.status(404).json({ error: "Usuario no encontrado" });
    res.status(200).json(actualizado);
  } catch (error) {
    next(error);
  }
};

module.exports = {
  addUsuario,
  loginUsuario,
  obtenerUsuarios,
  eliminarUsuario,
  modificarUsuario,
};
