import apiClient from "./apiClient";

export const getUserConversations = async (usuario_id) => {
  const { data } = await apiClient.post("/conversaciones", { usuario_id });
  return data;
};
