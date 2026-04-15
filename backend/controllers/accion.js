const { createAccion } = require("../data/accionData");

const addAccion = async (req, res, next) => {
  try {
    const { usuario_id, accion, current_exercise } = req.body;
    const data = await createAccion(usuario_id, accion, current_exercise, new Date());
    res.status(201).json(data);
  } catch (error) {
    next(error);
  }
};

module.exports = { addAccion };
