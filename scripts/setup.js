#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline/promises");
const crypto = require("node:crypto");
const { stdin, stdout } = require("node:process");

const ROOT = path.resolve(__dirname, "..");

const copyIfMissing = (src, dest) => {
  if (fs.existsSync(dest)) {
    console.log(`✔ ${path.relative(ROOT, dest)} ya existe`);
    return false;
  }
  fs.copyFileSync(src, dest);
  console.log(`✚ creado ${path.relative(ROOT, dest)}`);
  return true;
};

const replaceKey = (filePath, key, value) => {
  const content = fs.readFileSync(filePath, "utf8");
  const re = new RegExp(`^${key}=.*$`, "m");
  if (!re.test(content)) return;
  fs.writeFileSync(filePath, content.replace(re, `${key}=${value}`));
};

const main = async () => {
  console.log("\n🔧 Calculus-App — setup inicial\n");

  const backendEnv = path.join(ROOT, "backend/.env");
  const frontendEnv = path.join(ROOT, "frontend/.env");

  const createdBackend = copyIfMissing(path.join(ROOT, "backend/.env.example"), backendEnv);
  copyIfMissing(path.join(ROOT, "frontend/.env.example"), frontendEnv);

  if (createdBackend) {
    const secret = crypto.randomBytes(32).toString("hex");
    replaceKey(backendEnv, "JWT_SECRET", secret);
    console.log("✚ JWT_SECRET generado automáticamente");

    const rl = readline.createInterface({ input: stdin, output: stdout });
    const key = (await rl.question("\n¿Clave de OpenAI (sk-...)? [enter para omitir] ")).trim();
    rl.close();
    if (key) {
      replaceKey(backendEnv, "OPENAI_API_KEY", key);
      console.log("✚ OPENAI_API_KEY guardada en backend/.env");
    } else {
      console.log("⚠  No se configuró OPENAI_API_KEY — edita backend/.env antes de arrancar");
    }
  }

  console.log("\n✅ Listo. Próximos pasos:\n");
  console.log("   npm run db:up     # levanta Postgres en Docker");
  console.log("   npm run dev       # arranca backend + frontend\n");
  console.log("O, todo en uno:\n");
  console.log("   npm start\n");
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
