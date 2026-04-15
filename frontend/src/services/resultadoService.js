import apiClient from "./apiClient";

export const resultadoService = async (resultado_alumno, current_exercise) => {
  const user = JSON.parse(localStorage.getItem("user"));
  const usuario_id = user?.id;
  if (!usuario_id) throw new Error("Usuario no autenticado.");

  const { data } = await apiClient.post("/resultado", {
    usuario_id,
    resultado_alumno,
    current_exercise,
  });
  return data;
};

export const getResultados = async (usuario_id) => {
  const { data } = await apiClient.post("/resultados-usuario", { usuario_id });
  return data;
};
