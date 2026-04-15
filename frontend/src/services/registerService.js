import apiClient from "./apiClient";

export const register = async (nombre, correo, nickname, contrasena) => {
  const { data } = await apiClient.post("/usuario", {
    nombre,
    correo,
    nickname,
    contrasena,
  });
  return data;
};

export const getAllUsers = async () => {
  const { data } = await apiClient.get("/usuario");
  return data;
};

export const deleteUser = async (id) => {
  const { data } = await apiClient.delete(`/usuario/${id}`);
  return data;
};

export const editUser = async (id, updatedData) => {
  const { data } = await apiClient.put(`/usuario/${id}`, updatedData);
  return data;
};
