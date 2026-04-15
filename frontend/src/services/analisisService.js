import apiClient from "./apiClient";

export const analisisService = async () => {
  const user = JSON.parse(localStorage.getItem("user"));
  const usuario_id = user?.id;
  if (!usuario_id) throw new Error("Usuario no autenticado.");

  const { data } = await apiClient.post("/analisis", { usuario_id });
  return data;
};

export const getAnalisis = async (usuario_id) => {
  const { data } = await apiClient.post("/analisis2", { usuario_id });
  return data;
};
