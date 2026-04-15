require("dotenv").config();

const express = require("express");
const cors = require("cors");
const path = require("path");
const swaggerUi = require("swagger-ui-express");
const yaml = require("yamljs");
const OpenApiValidator = require("express-openapi-validator");

const { authRequired } = require("./middleware/auth");

const app = express();
const port = process.env.PORT || 3002;

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "http://localhost:5173")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);

app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      return callback(new Error(`Origen no permitido: ${origin}`));
    },
    credentials: true,
  })
);

app.use(express.json());

app.get("/health", (_req, res) => res.json({ status: "ok" }));

const swaggerDocument = yaml.load(path.join(__dirname, "api/openapi.yml"));

app.use("/api-docs", (req, res, next) => {
  const expected = process.env.DOCS_PASSWORD;
  if (!expected) return res.status(503).send("DOCS_PASSWORD no configurado");
  const b64 = (req.headers.authorization || "").split(" ")[1] || "";
  const [login, password] = Buffer.from(b64, "base64").toString().split(":");
  if (login === "admin" && password === expected) return next();
  res.set("WWW-Authenticate", 'Basic realm="API Docs"');
  res.status(401).send("Authentication required.");
});
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerDocument));

const PUBLIC_PATHS = new Set([
  "POST /login",
  "POST /admin",
  "POST /usuario",
  "GET /health",
]);

app.use((req, res, next) => {
  const key = `${req.method} ${req.path}`;
  if (PUBLIC_PATHS.has(key) || req.path.startsWith("/api-docs")) return next();
  return authRequired(req, res, next);
});

app.use(
  OpenApiValidator.middleware({
    apiSpec: path.join(__dirname, "api/openapi.yml"),
    validateRequests: true,
    validateResponses: false,
    operationHandlers: path.join(__dirname, "controllers"),
  })
);

app.use((err, req, res, _next) => {
  const status = err.status || err.statusCode || 500;
  const isServerError = status >= 500;
  if (isServerError) {
    console.error(`[${req.method} ${req.path}]`, err);
  }
  res.status(status).json({
    error: err.message || "Error interno",
    ...(err.errors ? { details: err.errors } : {}),
  });
});

app.listen(port, () => {
  console.log(`🚀 Backend escuchando en http://localhost:${port}`);
});
