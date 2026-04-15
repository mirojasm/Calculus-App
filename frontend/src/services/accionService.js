import apiClient from "./apiClient";

export const accionService = async (accion, current_exercise) => {
  const user = JSON.parse(localStorage.getItem("user"));
  const usuario_id = user?.id;
  if (!usuario_id) throw new Error("Usuario no autenticado.");

  const { data } = await apiClient.post("/accion", {
    usuario_id,
    accion,
    current_exercise,
  });
  return data;
};
