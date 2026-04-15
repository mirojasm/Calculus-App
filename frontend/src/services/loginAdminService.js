import apiClient from "./apiClient";

export const login = async (nickname, contrasena) => {
  const { data } = await apiClient.post("/admin", { nickname, contrasena });
  if (data?.token) {
    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));
  }
  return data;
};
