import apiClient from "./apiClient";

export const sendMessage = async (mensaje_usuario, conversationId, currentExercise) => {
  const user = JSON.parse(localStorage.getItem("user"));
  const usuario_id = user?.id;
  if (!usuario_id) throw new Error("Usuario no autenticado.");

  const { data } = await apiClient.post("/chat", {
    mensaje_usuario,
    conversationId,
    usuario_id,
    currentExercise,
  });
  return data;
};
