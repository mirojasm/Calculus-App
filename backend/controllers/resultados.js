const {
  createResultado,
  getResultadosPorUsuario,
} = require("../data/resultadosData");

const RESPUESTAS_CORRECTAS = {
  1: "C) \\( A = \\frac{40}{4 + \\pi},\\quad B = \\frac{20}{4 + \\pi} \\)",
  2: "A) \\( f(x) = x^3 - 3x^2 + 2 \\)",
};

const addResultado = async (req, res, next) => {
  try {
    const { usuario_id, resultado_alumno, current_exercise } = req.body;
    const resultado_correcto = RESPUESTAS_CORRECTAS[current_exercise] ?? null;
    const estado_ejercicio =
      resultado_correcto && resultado_alumno === resultado_correcto ? "bueno" : "malo";

    const data = await createResultado(
      usuario_id,
      resultado_alumno,
      resultado_correcto,
      estado_ejercicio,
      current_exercise,
      new Date()
    );
    res.status(201).json(data);
  } catch (error) {
    next(error);
  }
};

const getResultadosUsuario = async (req, res, next) => {
  try {
    const resultados = await getResultadosPorUsuario(req.body.usuario_id);
    res.status(200).json(resultados);
  } catch (error) {
    next(error);
  }
};

module.exports = { addResultado, getResultadosUsuario };
