import React, { useEffect, useState } from "react";
import {
  getAllUsers,
  deleteUser,
  editUser,
  register, // <-- importar register
} from "../../services/registerService";
import "bootstrap/dist/css/bootstrap.min.css";
import HeaderAdmin from '../../components/AdminHeader/Header.jsx';
import "./UsuariosPage.css";

const UsuariosPage = () => {
  const [usuarios, setUsuarios] = useState([]);
  const [editId, setEditId] = useState(null);
  const [editData, setEditData] = useState({});
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [newUser, setNewUser] = useState({
    nombre: "",
    correo: "",
    nickname: "",
    contrasena: "",
  });

  // Obtener todos los usuarios al montar
  useEffect(() => {
    fetchUsuarios();
  }, []);

  const fetchUsuarios = async () => {
    try {
      const data = await getAllUsers();
      setUsuarios(data);
    } catch (err) {
      setError("Error al cargar los usuarios.");
    }
  };

  const handleDelete = async (id) => {
    try {
      await deleteUser(id);
      setUsuarios(usuarios.filter((u) => u.id !== id));
      setSuccess("Usuario eliminado correctamente.");
      setTimeout(() => setSuccess(""), 2000);
    } catch (err) {
      setError("Error al eliminar el usuario.");
    }
  };

  const handleEditClick = (usuario) => {
    setEditId(usuario.id);
    setEditData({
      nombre: usuario.nombre,
      correo: usuario.correo,
      nickname: usuario.nickname,
      contrasena: usuario.contrasena || "",
    });
    setError("");
    setSuccess("");
  };

  const handleEditChange = (e) => {
    setEditData({ ...editData, [e.target.name]: e.target.value });
  };

  const handleEditSave = async (id) => {
    try {
      await editUser(id, editData);
      setEditId(null);
      setSuccess("Usuario editado correctamente.");
      fetchUsuarios();
      setTimeout(() => setSuccess(""), 2000);
    } catch (err) {
      setError("Error al editar el usuario.");
    }
  };

  const handleCancel = () => {
    setEditId(null);
    setEditData({});
    setError("");
    setSuccess("");
  };

  const handleNewUserChange = (e) => {
    setNewUser({ ...newUser, [e.target.name]: e.target.value });
  };

  const handleAddUser = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    try {
      await register(
        newUser.nombre,
        newUser.correo,
        newUser.nickname,
        newUser.contrasena
      );
      setSuccess("Usuario agregado correctamente.");
      setNewUser({
        nombre: "",
        correo: "",
        nickname: "",
        contrasena: "",
      });
      fetchUsuarios();
      setTimeout(() => setSuccess(""), 2000);
    } catch (err) {
      setError("Error al agregar el usuario.");
    }
  };

  return (
    <>
      <HeaderAdmin />
      <div className="container py-5" >
        <h2 className="mb-4">Gestión de Usuarios</h2>
        {/* Formulario para agregar usuario */}
        <form className="mb-4" onSubmit={handleAddUser}>
          <div className="row g-2 align-items-end">
            <div className="col">
              <input
                type="text"
                name="nombre"
                value={newUser.nombre}
                onChange={handleNewUserChange}
                className="form-control"
                placeholder="Nombre y Apellido"
                required
              />
            </div>
            <div className="col">
              <input
                type="email"
                name="correo"
                value={newUser.correo}
                onChange={handleNewUserChange}
                className="form-control"
                placeholder="Correo"
                required
              />
            </div>
            <div className="col">
              <input
                type="text"
                name="nickname"
                value={newUser.nickname}
                onChange={handleNewUserChange}
                className="form-control"
                placeholder="Username"
                required
              />
            </div>
            <div className="col">
              <input
                type="text" // <-- Cambiar de "password" a "text"
                name="contrasena"
                value={newUser.contrasena}
                onChange={handleNewUserChange}
                className="form-control"
                placeholder="Contraseña"
                required
              />
            </div>
            <div className="col-auto">
              <button type="submit" className="btn btn-success">
                Agregar Usuario
              </button>
            </div>
          </div>
        </form>
        {error && <div className="alert alert-danger">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}
        <table className="table table-bordered table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th>
              <th>Nombre y Apellido</th>
              <th>Correo</th>
              <th>Username</th>
              <th>Contraseña</th>
              <th>Acciones</th>
            </tr>
          </thead>
          <tbody>
            {usuarios.map((usuario) =>
              editId === usuario.id ? (
                <tr key={usuario.id}>
                  <td>{usuario.id}</td>
                  <td>
                    <input
                      type="text"
                      name="nombre"
                      value={editData.nombre}
                      onChange={handleEditChange}
                      className="form-control"
                    />
                  </td>
                  <td>
                    <input
                      type="email"
                      name="correo"
                      value={editData.correo}
                      onChange={handleEditChange}
                      className="form-control"
                    />
                  </td>
                  <td>
                    <input
                      type="text"
                      name="nickname"
                      value={editData.nickname}
                      onChange={handleEditChange}
                      className="form-control"
                    />
                  </td>
                  <td>
                    <input
                      type="text"
                      name="contrasena"
                      value={editData.contrasena}
                      onChange={handleEditChange}
                      className="form-control"
                    />
                  </td>
                  <td>
                    <button
                      className="btn btn-success btn-sm me-2"
                      onClick={() => handleEditSave(usuario.id)}
                    >
                      Guardar
                    </button>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={handleCancel}
                    >
                      Cancelar
                    </button>
                  </td>
                </tr>
              ) : (
                <tr key={usuario.id}>
                  <td>{usuario.id}</td>
                  <td>{usuario.nombre}</td>
                  <td>{usuario.correo}</td>
                  <td>{usuario.nickname}</td>
                  <td>{usuario.contraseña}</td> {/* Mostrar contraseña en texto plano */}
                  <td>
                    <button
                      className="btn btn-primary btn-sm me-2"
                      onClick={() => handleEditClick(usuario)}
                    >
                      Editar
                    </button>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() => handleDelete(usuario.id)}
                    >
                      Eliminar
                    </button>
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    </>
  );
};

export default UsuariosPage;
