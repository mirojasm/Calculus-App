const Responde = require("../services/responde");
const Verificador = require("../services/verificador");
const Provocador = require("../services/provocador");
const Filtro = require("../services/filtro");
const { createChat, getConversationTurns } = require("../data/chatData");

const isQuestion = (text) => {
  const clean = text.trim().toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  const interrogativas = [
    "que", "quien", "quienes", "cuando", "donde", "por que",
    "como", "cual", "cuales", "para que", "es", "hay", "tengo",
  ];
  return clean.endsWith("?") || interrogativas.some((w) => clean.startsWith(w));
};

const toConversation = (turns, newUserMessage) => {
  const history = turns.flatMap((t) => [
    { sender: "user", text: t.mensaje_usuario, timestamp: t.creado_en },
    { sender: "bot", text: t.respuesta_chatbot, timestamp: t.creado_en },
  ]);
  history.push({ sender: "user", text: newUserMessage, timestamp: new Date() });
  return history;
};

const chat = async (req, res, next) => {
  try {
    const { mensaje_usuario, conversationId, usuario_id, currentExercise } = req.body;

    const previousTurns = await getConversationTurns(usuario_id, currentExercise);
    const conversation = toConversation(previousTurns, mensaje_usuario);
    const userMessageCount = previousTurns.length + 1;

    const filtroOK = (await Filtro.filtrar(mensaje_usuario)).includes("true");

    let botResponse;
    let collaborationPassed = false;
    let PHASE_REACHED = false;

    if (!filtroOK) {
      botResponse = "Lo siento, no puedo ayudar con eso.";
    } else if (userMessageCount % 4 === 0 && !isQuestion(mensaje_usuario)) {
      botResponse = await Provocador.generateResponse(conversation, currentExercise);
    } else {
      botResponse = await Responde.generateResponse(conversation, currentExercise);

      if ([5, 10].includes(userMessageCount) || userMessageCount >= 14) {
        PHASE_REACHED = true;
      }

      if (userMessageCount >= 20) {
        const verificadorResponse = await Verificador.generateResponse(
          conversation,
          currentExercise
        );
        if (verificadorResponse.includes("ACUERDO_CLARO") || userMessageCount >= 30) {
          collaborationPassed = true;
        }
      }
    }

    const timestamp = new Date();
    const nextConversationId = conversationId || `conv_${Date.now()}`;

    await createChat(
      usuario_id,
      nextConversationId,
      mensaje_usuario,
      botResponse,
      currentExercise,
      timestamp
    );

    res.status(200).json({
      conversationId: nextConversationId,
      response: botResponse,
      usuario_id,
      timestamp,
      currentExercise,
      collaborationPassed,
      PHASE_REACHED,
      userMessageCount,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = { chat };
