const { getConversaciones } = require("../data/conversacionesData");
const { getAnalisis, createAnalisis } = require("../data/analisisData");
const { analyzeChatbotLog } = require("../services/analisis");

const Conversaciones = async (req, res, next) => {
  try {
    const data = await getConversaciones(req.body.usuario_id);
    res.status(200).json(data);
  } catch (error) {
    next(error);
  }
};

const Analisis = async (req, res, next) => {
  try {
    const { usuario_id } = req.body;
    const existing = await getAnalisis(usuario_id);
    if (existing) {
      return res.status(200).json({ analisis: existing.analisis });
    }

    const conversaciones = await getConversaciones(usuario_id);
    const botResponse = await analyzeChatbotLog(conversaciones, usuario_id);
    const analisisText = botResponse?.feedback ?? botResponse;
    const scores = botResponse?.scores ?? null;

    await createAnalisis(usuario_id, analisisText, scores, new Date());
    res.status(200).json({ analisis: analisisText });
  } catch (error) {
    next(error);
  }
};

const Analisis2 = async (req, res, next) => {
  try {
    const data = await getAnalisis(req.body.usuario_id);
    res.status(200).json({ analisis: data?.analisis ?? null });
  } catch (error) {
    next(error);
  }
};

module.exports = { Conversaciones, Analisis, Analisis2 };
