# FlowForge DAG Engine — Matt Pocock TypeScript Architecture

An enterprise-grade, type-safe autonomous workflow DAG (Directed Acyclic Graph) orchestration engine built strictly adhering to **Matt Pocock's Total TypeScript architectural patterns** and **Kahn's topological sort algorithm**.

![FlowForge DAG Canvas](./screenshots/flowforge_dag_canvas.png)

---

## 🏛️ Architecture & Type System

1. **Nominal Branded Types**:
   - Distinct branded types `WorkflowId`, `NodeId`, `RunId` preventing accidental string cross-assignment at compile time.
2. **Discriminated Unions**:
   - 6 strongly typed node kinds (`trigger`, `transform`, `inference`, `condition`, `action`, `join`) with distinct schemas.
3. **Exhaustive Type Narrowing**:
   - Compile-time exhaustiveness guarantees via `assertNever()`.
4. **Kahn's Acyclicity Algorithm**:
   - In-degree topological compiler detects circular dependencies and schedules parallel execution levels.
5. **Real-time SSE Streaming**:
   - Server-Sent Events event bus transmitting live telemetry.

![FlowForge TypeScript Lab](./screenshots/flowforge_typescript_lab.png)

---

## 🚀 Quick Start

```bash
# Backend (FastAPI on Port 8009)
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8009

# Frontend (Vite React on Port 5182)
cd frontend
npm install
npm run dev
```
