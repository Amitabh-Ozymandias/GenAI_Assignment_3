import "dotenv/config";
import express from "express";
import multer from "multer";
import { dirname, extname, join } from "path";
import { existsSync, mkdirSync, readFileSync, unlinkSync } from "fs";
import { fileURLToPath } from "url";
import { v4 as generateUuid } from "uuid";

// LangChain and LLM dependencies
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse/lib/pdf-parse.js");
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

// App constants
const AZURE_AI_BASE_URL = "https://models.inference.ai.azure.com";
const AI_API_KEY = process.env.GITHUB_TOKEN;
const EMBEDDING_MODEL = "text-embedding-3-large";
const CHAT_MODEL = "gpt-4o-mini";

// Initialization
const currentDir = dirname(fileURLToPath(import.meta.url));
const server = express();
server.use(express.json());
server.use(express.static(join(currentDir, "public")));

// Setup upload directory
const UPLOAD_FOLDER = "uploads";
if (!existsSync(UPLOAD_FOLDER)) {
  mkdirSync(UPLOAD_FOLDER, { recursive: true });
}
const fileUploader = multer({ dest: `${UPLOAD_FOLDER}/` });

const getVectorDbSettings = () => {
  const settings = { url: process.env.QDRANT_URL || "http://localhost:6333" };
  if (process.env.QDRANT_API_KEY) {
    settings.apiKey = process.env.QDRANT_API_KEY;
  }
  return settings;
};

// Handlers
const handleDocumentIngestion = async (req, res) => {
  const uploadedFile = req.file;
  if (!uploadedFile) {
    return res.status(400).json({ error: "File must be provided." });
  }

  const { path: tempFilePath, originalname: fileName } = uploadedFile;
  const fileExtension = extname(fileName).toLowerCase();

  if (![".pdf", ".txt"].includes(fileExtension)) {
    unlinkSync(tempFilePath);
    return res.status(400).json({ error: "Unsupported file type. Use PDF or TXT." });
  }

  const chatSessionId = generateUuid();

  try {
    let documentItems = [];

    // Extract text depending on file type
    if (fileExtension === ".pdf") {
      const fileBuffer = readFileSync(tempFilePath);
      const parsedOutput = await pdfParse(fileBuffer);
      console.log(`[Upload] PDF text length: ${parsedOutput.text.length}`);
      console.log(`[Upload] PDF text preview: "${parsedOutput.text.substring(0, 300)}..."`);
      documentItems = [{ pageContent: parsedOutput.text, metadata: { source: fileName } }];
    } else if (fileExtension === ".txt") {
      const fileText = readFileSync(tempFilePath, "utf-8");
      console.log(`[Upload] TXT text length: ${fileText.length}`);
      documentItems = [{ pageContent: fileText, metadata: { source: fileName } }];
    }

    // Split text into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const documentChunks = await textSplitter.splitDocuments(documentItems);
    console.log(`[Upload] Created ${documentChunks.length} chunks`);

    // Generate embeddings and store in Qdrant
    const embeddingService = new OpenAIEmbeddings({
      model: EMBEDDING_MODEL,
      apiKey: AI_API_KEY,
      configuration: { baseURL: AZURE_AI_BASE_URL },
    });

    console.log(`[Upload] Storing in Qdrant collection: ${chatSessionId}`);
    await QdrantVectorStore.fromDocuments(documentChunks, embeddingService, {
      ...getVectorDbSettings(),
      collectionName: chatSessionId,
    });
    console.log(`[Upload] Successfully stored in Qdrant`);

    unlinkSync(tempFilePath);
    return res.json({ sessionId: chatSessionId, chunks: documentChunks.length });
  } catch (error) {
    if (existsSync(tempFilePath)) {
      unlinkSync(tempFilePath);
    }
    console.error("Ingestion Error:", error);
    return res.status(500).json({ error: error.message });
  }
};

const handleChatRequest = async (req, res) => {
  const { question: userQuery, sessionId: chatSessionId } = req.body;

  if (!userQuery || !chatSessionId) {
    return res.status(400).json({ error: "Both 'question' and 'sessionId' are required fields." });
  }

  try {
    console.log(`[Chat] Session: ${chatSessionId}, Query: "${userQuery}"`);

    const embeddingService = new OpenAIEmbeddings({
      model: EMBEDDING_MODEL,
      apiKey: AI_API_KEY,
      configuration: { baseURL: AZURE_AI_BASE_URL },
    });

    const qdrantStore = await QdrantVectorStore.fromExistingCollection(embeddingService, {
      ...getVectorDbSettings(),
      collectionName: chatSessionId,
    });

    // Use similaritySearchWithScore to see actual scores
    const searchResults = await qdrantStore.similaritySearchWithScore(userQuery, 5);

    console.log(`[Chat] Retrieved ${searchResults.length} chunks`);
    searchResults.forEach(([doc, score], i) => {
      console.log(`  Chunk ${i + 1}: score=${score}, preview="${doc.pageContent.substring(0, 100)}..."`);
    });

    const relevantChunks = searchResults.map(([doc]) => doc);

    if (relevantChunks.length === 0) {
      console.log("[Chat] No chunks retrieved from Qdrant!");
      return res.json({ answer: "I couldn't find that in the document. (No chunks retrieved from vector store)" });
    }

    const contextString = relevantChunks
      .map((chunk, index) => {
        let header = `[Context ${index + 1}`;
        if (chunk.metadata?.loc?.pageNumber) {
          header += `, Page ${chunk.metadata.loc.pageNumber}`;
        }
        header += `]`;
        return `${header}\n${chunk.pageContent}`;
      })
      .join("\n\n---\n\n");

    console.log(`[Chat] Context length: ${contextString.length} chars`);

    const llmClient = new OpenAI({ baseURL: AZURE_AI_BASE_URL, apiKey: AI_API_KEY });

    const systemPrompt = `You are a helpful AI assistant. Answer the user's question based on the document context below. Use the information provided to give a thorough answer. If the context truly does not contain relevant information, state "I couldn't find that in the document."\n\nDocument context:\n${contextString}`;

    const completion = await llmClient.chat.completions.create({
      model: CHAT_MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userQuery },
      ],
    });

    console.log(`[Chat] LLM response: "${completion.choices[0].message.content.substring(0, 200)}..."`);

    return res.json({ answer: completion.choices[0].message.content });
  } catch (error) {
    console.error("Chat Error:", error);
    return res.status(500).json({ error: error.message });
  }
};

// Debug endpoint to check what's in Qdrant
server.get("/debug/:sessionId", async (req, res) => {
  try {
    const embeddingService = new OpenAIEmbeddings({
      model: EMBEDDING_MODEL,
      apiKey: AI_API_KEY,
      configuration: { baseURL: AZURE_AI_BASE_URL },
    });

    const qdrantStore = await QdrantVectorStore.fromExistingCollection(embeddingService, {
      ...getVectorDbSettings(),
      collectionName: req.params.sessionId,
    });

    const results = await qdrantStore.similaritySearch("test", 3);
    res.json({
      sessionId: req.params.sessionId,
      chunksFound: results.length,
      chunks: results.map((doc, i) => ({
        index: i,
        contentPreview: doc.pageContent.substring(0, 200),
        metadata: doc.metadata,
      })),
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Routes
server.post("/upload", fileUploader.single("file"), handleDocumentIngestion);
server.post("/chat", handleChatRequest);

// Start server
const SERVER_PORT = process.env.PORT || 3000;
server.listen(SERVER_PORT, () => {
  console.log(`Server is successfully running at http://localhost:${SERVER_PORT}`);
});
