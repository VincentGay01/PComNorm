﻿#define GLM_ENABLE_EXPERIMENTAL
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

std::string err, warn;
std::vector<float> texCoords;
std::vector<float> normals;
std::vector<int> texNumbers;
std::vector<std::string> textureFiles;
cv::Mat rvec, tvec;
bool poseFound = false;
bool modelbool = false;
// Variables globales pour la caméra
float objectRotationX = 0.0f;
float objectRotationY = 0.0f;
float cameraDistance = 1.0f; // Distance de la caméra par rapport à l'objet
float cameraAngleX = 0.0f;   // Angle de rotation autour de l'axe X
float cameraAngleY = 0.0f;   // Angle de rotation autour de l'axe Y
bool isDragging = false;     // Indique si la souris est en train de glisser
double lastMouseX, lastMouseY;
glm::vec3 cameraTarget = glm::vec3(0.0f); // Cible de la caméra (peut être déplacée)
float cameraSpeed = 0.001f;               // Vitesse de déplacement de la caméra
bool keys[1024] = { false };            // État des touches
glm::mat4 alignmentMatrix(1.0f); // Matrice d'alignement pour le plan d'overlay
std::vector<float> vertices; // Variables pour le maillage
std::vector<unsigned int> indices; // Indices du maillage
// Variables pour le plan d'overlay
GLuint overlayVAO, overlayVBO, overlayEBO;
GLuint overlayTexture;
GLuint overlayShaderProgram;
GLuint shaderProgram;
GLuint texCoordVBO;
GLuint textureID;
glm::vec3 cameraPos;
glm::mat4 view;
glm::mat4 newview;
float overlayZ = -0.5f; // Distance entre la caméra et le plan d'overlay 


// Matrices de vue et de projection
glm::mat4 projection = glm::perspective(glm::radians(60.f), 1.0f, 0.1f, 1000.0f);
glm::mat4 Model;
glm::mat4 MVP;
// Callback pour ajuster la taille de la fenêtre
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


// Callback pour gérer les entrées du clavier
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            keys[key] = true;
        }
        else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }
}





// Callback pour gérer les entrées de la souris
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

// Callback pour gérer le mouvement de la souris
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        lastMouseX = xpos;
        lastMouseY = ypos;

        // Ajuster les angles de rotation de l'objet
        objectRotationX += static_cast<float>(dy) * 0.1f;
        objectRotationY += static_cast<float>(dx) * 0.1f;
    }
}

// Callback pour gérer le zoom avec la molette
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cameraDistance -= static_cast<float>(yoffset) * 0.05f;
    if (cameraDistance < 0.001f) cameraDistance = 0.001f; // Limiter le zoom
    if (cameraDistance > 20.0f) cameraDistance = 20.0f; // Limiter le dézoom
}

// Fonction pour compiler un shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Vérification des erreurs de compilation
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Erreur de compilation du shader : " << infoLog << std::endl;
    }
    return shader;
}

// Fonction pour créer un programme shader
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Vérification des erreurs de liaison
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Erreur de liaison du programme shader : " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}





bool loadPLY(const std::string& filename, std::vector<float>& vertices, std::vector<unsigned int>& indices,
    std::vector<float>& normals, std::vector<float>& texCoords, std::vector<int>& texNumbers,
    std::vector<std::string>& textureFiles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }

    std::string line;
    bool headerEnded = false;
    int vertexCount = 0, faceCount = 0;
    bool hasNormals = false;
    bool hasTexCoords = false;
    bool hasTexNumber = false;
    bool hasTexCoordList = false;
    bool hasTextureFile = false;

    // Lecture de l'en-tête
    while (std::getline(file, line)) {
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word == "element") {
            std::string type;
            int count;
            iss >> type >> count;
            if (type == "vertex") vertexCount = count;
            if (type == "face") faceCount = count;
        }
        else if (word == "property" && line.find("nx") != std::string::npos) {
            hasNormals = true;
        }
        else if (word == "property" && (line.find("s") != std::string::npos ||
            line.find("u") != std::string::npos ||
            line.find("texture_u") != std::string::npos)) {
            hasTexCoords = true;
        }
        else if (word == "property" && line.find("texnumber") != std::string::npos) {
            hasTexNumber = true;
        }
        else if (word == "property" && line.find("list") != std::string::npos && line.find("texcoord") != std::string::npos) {
            hasTexCoordList = true;
        }
        else if (word == "comment" && line.find("TextureFile") != std::string::npos) {
            // Extraction du nom du fichier de texture depuis le commentaire
            size_t pos = line.find("TextureFile");
            if (pos != std::string::npos) {
                std::string texturePath = line.substr(pos + 11); // 11 est la longueur de "TextureFile"
                // Supprimer les espaces en début et fin
                texturePath.erase(0, texturePath.find_first_not_of(" \t"));
                texturePath.erase(texturePath.find_last_not_of(" \t") + 1);
                textureFiles.push_back(texturePath);
                hasTextureFile = true;
            }
        }
    }

    if (!headerEnded) {
        std::cerr << "Erreur : En-tête du fichier .ply incorrect" << std::endl;
        return false;
    }

    // Variables temporaires pour stocker les positions
    std::vector<glm::vec3> positions;

    // Lecture des sommets
    for (int i = 0; i < vertexCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        float x, y, z;
        iss >> x >> y >> z;

        // Stocker la position pour le calcul des normales si nécessaire
        positions.push_back(glm::vec3(x, y, z));

        // Ajouter la position aux vertices
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);

        if (hasNormals) {
            // Si le fichier contient des normales, les lire
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        }

        if (hasTexCoords && !hasTexCoordList) {
            // Si le fichier contient des coordonnées de texture standard, les lire
            float u, v;
            iss >> u >> v;
            texCoords.push_back(u);
            texCoords.push_back(v);
        }

        if (hasTexNumber) {
            // Si le fichier contient un numéro de texture, le lire
            int texNum;
            iss >> texNum;
            texNumbers.push_back(texNum);
        }
    }

    // Stocker les triangles pour calculer les normales si nécessaire
    std::vector<glm::uvec3> triangles;

    // Lecture des faces
    for (int i = 0; i < faceCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);

        if (hasTexCoordList) {
            // Format: nombre_vertices v1 v2 v3 nombre_coords_tex tx1 ty1 tx2 ty2 tx3 ty3
            int vertexCount, v1, v2, v3;
            iss >> vertexCount >> v1 >> v2 >> v3;

            if (vertexCount == 3) { // On ne gère que les triangles
                indices.push_back(v1);
                indices.push_back(v2);
                indices.push_back(v3);

                // Stocker le triangle pour le calcul des normales
                triangles.push_back(glm::uvec3(v1, v2, v3));

                // Lire le nombre de coordonnées de texture
                unsigned char numTexCoords;
                iss >> numTexCoords;

                // Lire les coordonnées de texture pour chaque sommet du triangle
                for (int j = 0; j < numTexCoords && j < vertexCount * 2; j += 2) {
                    float u, v;
                    iss >> u >> v;

                    // Assurez-vous que texCoords a suffisamment d'espace
                    while (texCoords.size() <= v1 * 2 + j + 1) {
                        texCoords.push_back(0.0f);
                        texCoords.push_back(0.0f);
                    }

                    // Stocker les coordonnées de texture
                    if (j == 0) {
                        texCoords[v1 * 2] = u;
                        texCoords[v1 * 2 + 1] = v;
                    }
                    else if (j == 2) {
                        texCoords[v2 * 2] = u;
                        texCoords[v2 * 2 + 1] = v;
                    }
                    else if (j == 4) {
                        texCoords[v3 * 2] = u;
                        texCoords[v3 * 2 + 1] = v;
                    }
                }
            }
        }
        else {
            // Format standard: nombre_vertices v1 v2 v3
            int vertexCount, v1, v2, v3;
            iss >> vertexCount >> v1 >> v2 >> v3;

            if (vertexCount == 3) { // On ne gère que les triangles
                indices.push_back(v1);
                indices.push_back(v2);
                indices.push_back(v3);

                // Stocker le triangle pour le calcul des normales
                triangles.push_back(glm::uvec3(v1, v2, v3));
            }
        }
    }

    // Si le fichier ne contient pas de normales, les calculer
    if (!hasNormals) {
        // Initialiser les normales à zéro
        normals.resize(positions.size() * 3, 0.0f);

        // Calculer les normales pour chaque triangle
        for (const auto& triangle : triangles) {
            glm::vec3 v0 = positions[triangle[0]];
            glm::vec3 v1 = positions[triangle[1]];
            glm::vec3 v2 = positions[triangle[2]];

            // Calculer les vecteurs des côtés du triangle
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;

            // Calculer la normale du triangle par produit vectoriel
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

            // Ajouter cette normale aux trois sommets du triangle
            for (int j = 0; j < 3; ++j) {
                int idx = triangle[j] * 3;
                normals[idx] += normal.x;
                normals[idx + 1] += normal.y;
                normals[idx + 2] += normal.z;
            }
        }

        // Normaliser toutes les normales
        for (size_t i = 0; i < normals.size(); i += 3) {
            glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
            n = glm::normalize(n);
            normals[i] = n.x;
            normals[i + 1] = n.y;
            normals[i + 2] = n.z;
        }
    }

    // Si le fichier ne contient pas de coordonnées de texture, initialiser avec des valeurs par défaut
    if (!hasTexCoords && !hasTexCoordList) {
        texCoords.resize(positions.size() * 2, 0.0f);
    }

    return true;
}


bool loadPLY_clean(const std::string& filename,
    std::vector<float>& vertices,
    std::vector<unsigned int>& indices,
    std::vector<float>& normals,
    std::vector<float>& texCoords,
    std::vector<std::string>& textureFiles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }

    std::string line;
    bool headerEnded = false;
    int vertexCount = 0, faceCount = 0;
    bool hasNormals = false;
    bool hasTexCoordList = false;

    // Temporaire pour lire les sommets et normales
    std::vector<glm::vec3> tempPositions;
    std::vector<glm::vec3> tempNormals;

    // Lire header
    while (std::getline(file, line)) {
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word == "element") {
            std::string type;
            int count;
            iss >> type >> count;
            if (type == "vertex") vertexCount = count;
            if (type == "face") faceCount = count;
        }
        else if (word == "property" && line.find("nx") != std::string::npos) {
            hasNormals = true;
        }
        else if (word == "property list" && line.find("texcoord") != std::string::npos) {
            hasTexCoordList = true;
        }
        else if (word == "comment" && line.find("TextureFile") != std::string::npos) {
            size_t pos = line.find("TextureFile");
            if (pos != std::string::npos) {
                std::string texturePath = line.substr(pos + 11);
                texturePath.erase(0, texturePath.find_first_not_of(" \t"));
                texturePath.erase(texturePath.find_last_not_of(" \t") + 1);
                textureFiles.push_back(texturePath);
            }
        }
    }

    if (!headerEnded) {
        std::cerr << "Erreur : fin de header non trouvée" << std::endl;
        return false;
    }

    // Lire les vertices
    for (int i = 0; i < vertexCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        float x, y, z, nx = 0.0f, ny = 0.0f, nz = 0.0f;
        iss >> x >> y >> z;
        if (hasNormals) {
            iss >> nx >> ny >> nz;
        }
        tempPositions.emplace_back(x, y, z);
        tempNormals.emplace_back(nx, ny, nz);
    }

    // Lire les faces avec texcoord list
    for (int i = 0; i < faceCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);

        int nVerts;
        iss >> nVerts;
        if (nVerts != 3) continue; // on ignore les non-triangles

        int i0, i1, i2;
        iss >> i0 >> i1 >> i2;

        int texCoordCount;
        iss >> texCoordCount;

        std::vector<float> tex;
        for (int j = 0; j < texCoordCount; ++j) {
            float val;
            iss >> val;
            tex.push_back(val);
        }

        // Chaque triangle → 3 sommets (non partagés)
        glm::ivec3 ids = { i0, i1, i2 };

        for (int j = 0; j < 3; ++j) {
            glm::vec3 pos = tempPositions[ids[j]];
            glm::vec3 norm = tempNormals[ids[j]];
            float u = tex[j * 2];
            float v = tex[j * 2 + 1];

            vertices.push_back(pos.x);
            vertices.push_back(pos.y);
            vertices.push_back(pos.z);

            normals.push_back(norm.x);
            normals.push_back(norm.y);
            normals.push_back(norm.z);

            texCoords.push_back(u);
            texCoords.push_back(1.0f - v);

            indices.push_back(indices.size());
        }
    }

    return true;
}


// Fonction pour projeter les sommets sur l'image
std::vector<cv::Point2f> projectVerticesToImage(
    const std::vector<float>& vertices,
    const glm::mat4& MVP,
    int image_width,
    int image_height)
{
    std::vector<cv::Point2f> projectedPoints;

    for (size_t i = 0; i < vertices.size(); i += 3) {
        glm::vec4 vertex(vertices[i], vertices[i + 1], vertices[i + 2], 1.0f);
        glm::vec4 clipSpace = MVP * vertex;

        if (clipSpace.w == 0.0f) continue; // éviter division par zéro

        glm::vec3 ndc = glm::vec3(clipSpace) / clipSpace.w; // Normalized Device Coordinates [-1,1]

        float x_img = (ndc.x * 0.5f + 0.5f) * image_width;
        float y_img = (1.0f - (ndc.y * 0.5f + 0.5f)) * image_height; // inverser Y pour OpenCV

        projectedPoints.push_back(cv::Point2f(x_img, y_img));
    }

    return projectedPoints;
}

std::vector<std::pair<cv::Point2f, glm::vec3>> matchKeypointsTo3D(
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::Point2f>& projectedVertices,
    const std::vector<float>& vertices3D,
    float maxDistancePixels = 1.0f) // Réduire le seuil pour plus de précision
{
    std::vector<std::pair<cv::Point2f, glm::vec3>> matches;

    if (projectedVertices.empty() || vertices3D.empty()) return matches;

    // Construire la matrice de recherche pour FLANN
    cv::Mat projectedMat(projectedVertices.size(), 2, CV_32F);
    for (size_t i = 0; i < projectedVertices.size(); ++i) {
        projectedMat.at<float>(i, 0) = projectedVertices[i].x;
        projectedMat.at<float>(i, 1) = projectedVertices[i].y;
    }

    // Création du KD-Tree FLANN
    cv::flann::KDTreeIndexParams indexParams(6); // 5 arbres
    projectedMat.convertTo(projectedMat, CV_32F);
    cv::flann::Index kdtree(projectedMat, indexParams);

    for (const auto& kp : keypoints) {
        cv::Mat query(1, 2, CV_32F);
        query.at<float>(0) = kp.pt.x;
        query.at<float>(1) = kp.pt.y;

        std::vector<int> indices(1);
        std::vector<float> dists(1);

        kdtree.knnSearch(query, indices, dists, 1, cv::flann::SearchParams(32));

        if (!indices.empty() && dists[0] < maxDistancePixels * maxDistancePixels) {
            int idx = indices[0];

            glm::vec3 corresponding3D(
                vertices3D[3 * idx],
                vertices3D[3 * idx + 1],
                vertices3D[3 * idx + 2]
            );

            matches.emplace_back(kp.pt, corresponding3D);
        }
    }

    // Ajouter un filtre supplémentaire pour rejeter les correspondances douteuses
    std::vector<std::pair<cv::Point2f, glm::vec3>> filteredMatches;
    for (const auto& match : matches) {
        // Vous pourriez ajouter des vérifications supplémentaires ici
        // Par exemple, vérifier si la profondeur Z est raisonnable
        if (match.second.z > -10.0f && match.second.z < 10.0f) {
            filteredMatches.push_back(match);
        }
    }

    return filteredMatches;
}


// fonction Pnp pour estimer pose de la cam
bool estimateCameraPoseFromMatches(
    const std::vector<std::pair<cv::Point2f, glm::vec3>>& matches,
    const cv::Mat& cameraMatrix,
    cv::Mat& rvec,
    cv::Mat& tvec,
    bool useExtrinsicGuess = true)
{
    if (matches.size() < 4) {
        std::cerr << "Pas assez de correspondances pour solvePnP (minimum 4 requis)" << std::endl;
        return false;
    }

    // Convertir les correspondances en vecteurs pour solvePnP
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (const auto& match : matches) {
        imagePoints.push_back(match.first); // 2D
        objectPoints.emplace_back(
            match.second.x,
            match.second.y,
            match.second.z
        ); // 3D
    }

    // Aucune distorsion supposée (ou tu peux la passer en paramètre si tu veux)
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    bool success = cv::solvePnPRansac(
        objectPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        useExtrinsicGuess,
        100000,           // Augmenter le nombre d'itérations
        2.0,            // Seuil de reprojection
        0.99,           // Niveau de confiance
        cv::noArray(),  // Inliers
        cv::SOLVEPNP_SQPNP  
    );

    // Raffiner la pose si la résolution initiale a réussi
    if (success) {
        cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix,
            distCoeffs, rvec, tvec);
    }

    return success;
}

glm::mat4 createViewMatrixFromPnP(const cv::Mat& rvec, const cv::Mat& tvec)
{
    // Convertir le vecteur de rotation en matrice de rotation
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Créer une matrice de transformation OpenGL
    glm::mat4 viewMatrix = glm::mat4(1.0f);

    // Remplir la matrice de rotation (transposée pour OpenGL)
    // OpenCV: X droite, Y bas, Z avant
    // OpenGL: X droite, Y haut, Z arrière
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            viewMatrix[i][j] = R.at<double>(j, i);
        }
    }

    // Inverser Y et Z pour la conversion OpenCV -> OpenGL
    viewMatrix[1][0] *= -1; viewMatrix[1][1] *= -1; viewMatrix[1][2] *= -1;
    viewMatrix[2][0] *= -1; viewMatrix[2][1] *= -1; viewMatrix[2][2] *= -1;

    // Appliquer la translation
    viewMatrix[3][0] = tvec.at<double>(0);
    viewMatrix[3][1] = -tvec.at<double>(1); // Inverser Y
    viewMatrix[3][2] = -tvec.at<double>(2); // Inverser Z

    // En OpenGL, nous voulons l'inverse de cette matrice pour la vue
    return glm::inverse(viewMatrix);
}


//fonction pour prendre photo de la scène
cv::Mat renderMeshToImage(const std::vector<float>& vertices,
    const std::vector<unsigned int>& indices,
    const std::vector<float>& normals,
    const std::vector<float>& texCoords, 
    GLuint shaderProgram,
    GLuint textureID,                    
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::mat4& model,
    int width = 2704,
    int height = 1800) {

    // Sauvegarder l'état actuel de OpenGL
    GLint previousFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFBO);
    GLint previousViewport[4];
    glGetIntegerv(GL_VIEWPORT, previousViewport);

    // Créer un FBO (Frame Buffer Object)
    GLuint fbo, renderTexture, depthBuffer;

    // Initialiser le FBO
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Créer une texture pour stocker le rendu
    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    // Paramètres de texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);

    // Créer un renderbuffer pour la profondeur
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

    // Vérifier si le FBO est complet
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Erreur lors de la création du FBO" << std::endl;

        // Nettoyer les ressources en cas d'erreur
        glBindFramebuffer(GL_FRAMEBUFFER, previousFBO);
        glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
        glDeleteTextures(1, &renderTexture);
        glDeleteRenderbuffers(1, &depthBuffer);
        glDeleteFramebuffers(1, &fbo);

        return cv::Mat();
    }

    // Configurer la vue de rendu
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Même couleur d'arrière-plan que l'écran principal
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activer le test de profondeur si ce n'est pas déjà fait
    GLboolean depthTestWasEnabled;
    glGetBooleanv(GL_DEPTH_TEST, &depthTestWasEnabled);
    if (!depthTestWasEnabled) {
        glEnable(GL_DEPTH_TEST);
    }

    // Créer un VAO et VBO temporaires pour ce rendu
    GLuint tempVAO, tempVBO, tempNormalVBO, tempEBO, tempTexCoordVBO;
    glGenVertexArrays(1, &tempVAO);
    glGenBuffers(1, &tempVBO);
    glGenBuffers(1, &tempNormalVBO); 
    glGenBuffers(1, &tempTexCoordVBO);   
    glGenBuffers(1, &tempEBO);

    glBindVertexArray(tempVAO);

    // Buffer pour les positions
    glBindBuffer(GL_ARRAY_BUFFER, tempVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer pour les normales
    glBindBuffer(GL_ARRAY_BUFFER, tempNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    //Buffer pour les texCoords
    glBindBuffer(GL_ARRAY_BUFFER, tempTexCoordVBO);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    // Buffer pour les indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Utiliser le shader et configurer ses uniformes
    glUseProgram(shaderProgram);

    // Utiliser EXACTEMENT le même modèle et matrices que dans l'affichage principal
    glm::mat4 MVP = projection * view * model;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "Model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Définir les mêmes paramètres d'éclairage que ceux utilisés pour l'affichage principal
    glm::vec3 lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor = glm::vec3(0.7f, 0.7f, 0.7f);
    float ambientStrength = 0.3f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos)); // Position globale de la caméra
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
    glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));
    glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), ambientStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), specularStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), shininess);

    // Dessiner le maillage
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 1); // 'texture1' dans ton shader

    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    // Lire les pixels du framebuffer
    std::vector<unsigned char> pixels(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Convertir en image OpenCV
    cv::Mat result = cv::Mat(height, width, CV_8UC3);

    // OpenGL rend du bas vers le haut, OpenCV stocke du haut vers le bas
    // Donc nous devons inverser verticalement l'image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int glIndex = (y * width + x) * 3;
            // Inverser l'axe Y et convertir RGB à BGR (OpenCV utilise BGR)
            result.at<cv::Vec3b>(height - 1 - y, x)[0] = pixels[glIndex + 2]; // B
            result.at<cv::Vec3b>(height - 1 - y, x)[1] = pixels[glIndex + 1]; // G
            result.at<cv::Vec3b>(height - 1 - y, x)[2] = pixels[glIndex + 0]; // R
        }
    }

    // Nettoyer les ressources
    glDeleteVertexArrays(1, &tempVAO);
    glDeleteBuffers(1, &tempVBO);
    glDeleteBuffers(1, &tempNormalVBO);
    glDeleteBuffers(1, &tempEBO);
    glDeleteBuffers(1, &tempTexCoordVBO);
    glDeleteTextures(1, &renderTexture);
    glDeleteRenderbuffers(1, &depthBuffer);
    glDeleteFramebuffers(1, &fbo);

    // Restaurer l'état précédent d'OpenGL
    glBindFramebuffer(GL_FRAMEBUFFER, previousFBO);
    glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);

    if (!depthTestWasEnabled) {
        glDisable(GL_DEPTH_TEST);
    }

    return result;
}



void alignByContoursRANSAC(const cv::Mat& img1, const cv::Mat& img2) {
    // 1. Détection de bords (Canny)
    cv::Mat edges1, edges2;
    cv::Canny(img1, edges1, 100, 200);
    cv::Canny(img2, edges2, 100, 200);

    // 2. Trouver les contours
    std::vector<std::vector<cv::Point>> contours1, contours2;
    cv::findContours(edges1, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(edges2, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3. Fusionner tous les points de contours en un seul nuage de points
    std::vector<cv::Point> points1, points2;
    for (auto& contour : contours1) points1.insert(points1.end(), contour.begin(), contour.end());
    for (auto& contour : contours2) points2.insert(points2.end(), contour.begin(), contour.end());

    // 4. Downsampling (on prend 1 point sur N pour accélérer)
    const int sampleRate = 10;
    std::vector<cv::Point2f> sampled1, sampled2;
    for (size_t i = 0; i < points1.size(); i += sampleRate) sampled1.push_back(points1[i]);
    for (size_t i = 0; i < points2.size(); i += sampleRate) sampled2.push_back(points2[i]);

    // 5. Matching naïf basé sur la distance (greedy matching)
    std::vector<cv::Point2f> matched1, matched2;
    for (auto& p1 : sampled1) {
        double minDist = 1e9;
        cv::Point2f bestMatch;
        for (auto& p2 : sampled2) {
            double dist = cv::norm(p1 - p2);
            if (dist < minDist) {
                minDist = dist;
                bestMatch = p2;
            }
        }
        matched1.push_back(p1);
        matched2.push_back(bestMatch);
    }

    // 6. Estimer la transformation avec RANSAC
    cv::Mat inlierMask;
    cv::Mat affine = cv::estimateAffinePartial2D(matched1, matched2, inlierMask, cv::RANSAC);

    if (affine.empty()) {
        std::cerr << "Erreur : pas de transformation trouvée !" << std::endl;
        return;
    }

    // 7. Appliquer la transformation
    cv::Mat aligned;
    cv::warpAffine(img1, aligned, affine, img2.size());

    // 8. Afficher les résultats
    cv::imwrite("Image 1 (Originale)", img1);
    cv::imwrite("Image 2 (Reference)", img2);
    cv::imwrite("Image 1 alignée", aligned);

    cv::waitKey(0);
}

// Fonction modifiée pour prendre en compte les matrices de vue et de projection actuelles
bool alignMeshWithImage(const char* imagePath,
    const std::vector<float>& vertices,
    const std::vector<unsigned int>& indices,
    const std::vector<float>& normals, // Ajout des normales
    GLuint shaderProgram,
     glm::mat4& view,
    const glm::mat4& projection,
    const glm::mat4& model,
    glm::mat4& outModelMatrix) {

    // 1. Charger l'image de référence
    cv::Mat referenceImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (referenceImage.empty()) {
        std::cerr << "Erreur: Impossible de charger l'image de référence: " << imagePath << std::endl;
        return false;
    }

    // 2. Préparer une image de rendu du maillage 3D avec la vue de caméra actuelle
    cv::Mat meshRendering = renderMeshToImage(vertices, indices, normals, texCoords,
        shaderProgram,textureID ,view, projection, model,
        referenceImage.cols, referenceImage.rows);

	
    // Optionnel: Sauvegarder les images pour débogage
    cv::imwrite("reference_image.png", referenceImage);
    cv::imwrite("mesh_rendering.png", meshRendering);

    // 3. Détecter des points caractéristiques dans les deux images
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create(
        20000,        // Nombre de keypoints maximum (par image)
        5,    // Nombre de couches par octave
        0.09,// Seuil de contraste (rejette points peu significatifs)
        20,    // Seuil pour éliminer les points près des bords
        1.2             // Écart-type du flou gaussien appliqué en entrée
    );
    std::vector<cv::KeyPoint> keypointsRef, keypointsMesh;
    cv::Mat descriptorsRef, descriptorsMesh;
    
    detector->detectAndCompute(referenceImage, cv::noArray(), keypointsRef, descriptorsRef);
    detector->detectAndCompute(meshRendering, cv::noArray(), keypointsMesh, descriptorsMesh);

    // Vérifier si des points caractéristiques ont été trouvés
    if (keypointsRef.empty() || keypointsMesh.empty()) {
        std::cerr << "Erreur: Aucun point caractéristique détecté dans les images" << std::endl;
        return false;
    }

    if (descriptorsRef.empty() || descriptorsMesh.empty()) {
        std::cerr << "Erreur: Pas de descripteurs générés pour les points caractéristiques" << std::endl;
        return false;
    }

    std::cout << "Points caractéristiques détectés: " << keypointsRef.size()
        << " dans l'image de référence, " << keypointsMesh.size()
        << " dans le rendu du maillage." << std::endl;

    // 4. Matcher les points caractéristiques
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;

    // Vérifier si les descripteurs sont du bon type pour FLANN
    if (descriptorsRef.type() != CV_32F) {
        descriptorsRef.convertTo(descriptorsRef, CV_32F);
    }
    if (descriptorsMesh.type() != CV_32F) {
        descriptorsMesh.convertTo(descriptorsMesh, CV_32F);
    }

    try {
        matcher->knnMatch(descriptorsRef, descriptorsMesh, knnMatches, 2);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Erreur lors du matching: " << e.what() << std::endl;
        return false;
    }

    // 5. Filtrer les correspondances de bonne qualité
    const float ratioThresh = 0.7f;
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() < 2) continue; // Ignorer les matchs incomplets

        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // 6. Filtrer pour garantir une correspondance unique
       // Pour chaque keypoint, ne conserver que la meilleure correspondance
    std::map<int, cv::DMatch> bestMatchForRefKeypoint;    // queryIdx -> meilleur match
    std::map<int, cv::DMatch> bestMatchForMeshKeypoint;   // trainIdx -> meilleur match

    // Trouver les meilleures correspondances pour chaque keypoint (dans les deux directions)

    for (const auto& match : goodMatches) {
        // Pour les keypoints de référence
        auto refIt = bestMatchForRefKeypoint.find(match.queryIdx);
        if (refIt == bestMatchForRefKeypoint.end() || match.distance < refIt->second.distance) {
            bestMatchForRefKeypoint[match.queryIdx] = match;
        }

        // Pour les keypoints du maillage
        auto meshIt = bestMatchForMeshKeypoint.find(match.trainIdx);
        if (meshIt == bestMatchForMeshKeypoint.end() || match.distance < meshIt->second.distance) {
            bestMatchForMeshKeypoint[match.trainIdx] = match;
        }
    }

    // Garder uniquement les correspondances mutuelles (un keypoint de l'image 1 est associé à un keypoint 
    // de l'image 2 et vice versa)
    std::vector<cv::DMatch> uniqueMatches;
    for (const auto& pair : bestMatchForRefKeypoint) {
        const cv::DMatch& match = pair.second;
        auto it = bestMatchForMeshKeypoint.find(match.trainIdx);

        // Vérifier si cette correspondance est mutuelle
        if (it != bestMatchForMeshKeypoint.end() &&
            it->second.queryIdx == match.queryIdx) {
            uniqueMatches.push_back(match);
        }
    }

    std::cout << "Nombre de correspondances initiales: " << goodMatches.size() << std::endl;
    std::cout << "Nombre de correspondances uniques: " << uniqueMatches.size() << std::endl;

    if (uniqueMatches.size() < 8) {
        std::cerr << "Pas assez de bonnes correspondances uniques trouvées (minimum 8 requises)" << std::endl;
        return false;
    }

	// fonction pour le calcul de la matrice MVP
    int image_width = referenceImage.cols;
    int image_height = referenceImage.rows;

    // 1. Projetter les sommets sur l'image
    glm::mat4 tempMVP = projection * view * model;
    std::vector<cv::Point2f> projectedVertices = projectVerticesToImage(vertices, tempMVP, image_width, image_height);

    // 2. Matcher les keypoints détectés sur l'image du mesh
    std::vector<std::pair<cv::Point2f, glm::vec3>> matchedPoints = matchKeypointsTo3D(keypointsMesh, projectedVertices, vertices);

    // 3. Utiliser les correspondances
    cv::Mat debugImage = meshRendering.clone();
    for (const auto& m : matchedPoints) {
        cv::circle(debugImage, m.first, 4, cv::Scalar(0, 255, 0), -1);
    }
    cv::imwrite("debug_matched_points.png", debugImage);
    for (const auto& match : matchedPoints) {
        const cv::Point2f& keypoint2D = match.first;
        const glm::vec3& point3D = match.second;
    }

    //-----------------------------------------------------------------------------

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        2252.308217738484, 0, 1352.0,
        0, 2252.308217738484, 900.0,
        0, 0, 1);
  
    
    //  Appeler la fonction
    poseFound = estimateCameraPoseFromMatches(matchedPoints, cameraMatrix, rvec, tvec);

    if (poseFound) {
        std::cout << "Rotation (Rodrigues) : " << rvec.t() << std::endl;
        std::cout << "Translation : " << tvec.t() << std::endl;

        // Optionnel : convertir rvec en matrice de rotation
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        std::cout << "Rotation matrix : " << R << std::endl;
        
            newview = createViewMatrixFromPnP(rvec, tvec);

    }

    //-------------------------------------------------------------------
	// 5. Afficher les correspondances
    cv::Mat imgMatches;
    cv::drawMatches(referenceImage, keypointsRef, meshRendering, keypointsMesh,
        uniqueMatches, imgMatches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("matches.png", imgMatches);

    // 6. Extraire les points correspondants
    std::vector<cv::Point2f> pointsRef, pointsMesh;
    for (size_t i = 0; i < uniqueMatches.size(); i++) {
        pointsRef.push_back(keypointsRef[uniqueMatches[i].queryIdx].pt);
        pointsMesh.push_back(keypointsMesh[uniqueMatches[i].trainIdx].pt);
    }
    return true;
}





// Modification de la fonction processInput pour utiliser la vue de caméra actuelle
void processInput() {
    // Définir des vecteurs d'axes mondiaux
    glm::vec3 worldRight = glm::vec3(1.0f, 0.0f, 0.0f);   // Axe X
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);      // Axe Y
    glm::vec3 worldForward = glm::vec3(0.0f, 0.0f, -1.0f); // Axe Z (négatif car OpenGL)
    if (!poseFound) {


        if (keys[GLFW_KEY_W] || keys[GLFW_KEY_Z])
            cameraTarget += worldForward * cameraSpeed;
        if (keys[GLFW_KEY_S])
            cameraTarget -= worldForward * cameraSpeed;
        if (keys[GLFW_KEY_A] || keys[GLFW_KEY_Q])
            cameraTarget -= worldRight * cameraSpeed;
        if (keys[GLFW_KEY_D])
            cameraTarget += worldRight * cameraSpeed;
        if (keys[GLFW_KEY_SPACE])
            cameraTarget += worldUp * cameraSpeed;
        if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL])
            cameraTarget -= worldUp * cameraSpeed;
    }
    if (keys[GLFW_KEY_E]) {
        
        bool alignmentSuccess = alignMeshWithImage(
            "D:/project/PComNorm/shoe1.png", // Chemin de l'image de référence
            vertices,                            // Vos vertices
            indices,                             // Vos indices
            normals,                             // Vos normales
            shaderProgram,                       // Votre programme de shader
            view,                                // Vue actuelle de la caméra
            projection,                          // Projection actuelle
            Model,                               // Matrice modèle actuelle
            alignmentMatrix                      // Matrice de sortie
        );
    }

    // Déplacement haut/bas sur l'axe Y
    if (keys[GLFW_KEY_SPACE])
        cameraTarget += worldUp * cameraSpeed;
    if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL])
        cameraTarget -= worldUp * cameraSpeed;

    // Réinitialiser la position de la caméra
    if (keys[GLFW_KEY_R]) {
        cameraTarget = glm::vec3(0.0f);
        cameraAngleX = 0.0f;
        cameraAngleY = 0.0f;
        cameraDistance = 5.0f;
    }
}


int main() {
    // Initialisation de GLFW
    if (!glfwInit()) {
        std::cerr << "Échec de l'initialisation de GLFW" << std::endl;
        return -1;
    }

    // Configuration de la version OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Création de la fenêtre
    GLFWwindow* window = glfwCreateWindow(2000,2000, "Fenêtre OpenGL", nullptr, nullptr);
    if (!window) {
        std::cerr << "Échec de la création de la fenêtre GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialisation de GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Échec de l'initialisation de GLEW" << std::endl;
        return -1;
    }

    // Définir la fonction de callback pour le redimensionnement
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    // Activer le blending pour la transparence
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Activer le test de profondeur
    glEnable(GL_DEPTH_TEST);
    /*
    if (!loadPLY("D:/project/PComNorm/Chemi-AU-O0051.ply", vertices, indices, normals, texCoords, texNumbers, textureFiles)) {
        return -1;
    }
    */
    if (!loadPLY_clean("D:/project/PComNorm/Chemi-AU-O0052.ply", vertices, indices, normals, texCoords, textureFiles)) {
        return -1;
    }

    // Création du VAO, VBO et EBO avec support pour les normales
    GLuint VAO, VBO, normalVBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &normalVBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &texCoordVBO);
    glBindVertexArray(VAO);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Paramètres de texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    // Vérifier si nous avons des chemins de texture
    if (!textureFiles.empty()) {
        // Utiliser le premier fichier de texture trouvé dans le PLY
        std::string texturePath = "D:/project/PComNorm/" + textureFiles[0];  // Ajuster le chemin si nécessaire

        int texWidth, texHeight, texChannels;
        unsigned char* data = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, 0);
        if (data) {
            GLenum format = (texChannels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, texWidth, texHeight, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            stbi_image_free(data);  // Libérer la mémoire
            std::cout << "Texture chargée avec succès: " << texturePath << std::endl;
        }
        else {
            std::cerr << "Échec du chargement de la texture: " << texturePath << std::endl;
            std::cerr << "Erreur STBI: " << stbi_failure_reason() << std::endl;
        }
    }
    else {
        std::cerr << "Aucun fichier de texture trouvé dans le PLY" << std::endl;
    }
    

    // Afficher quelques coordonnées de texture pour débogage
    std::cout << "Nombre de coordonnées de texture: " << texCoords.size() / 2 << std::endl;
    if (!texCoords.empty()) {
        for (int i = 0; i < std::min(10, (int)texCoords.size() / 2); i++) {
            std::cout << "TexCoord[" << i << "]: (" << texCoords[i * 2] << ", " << texCoords[i * 2 + 1] << ")" << std::endl;
        }
    }

  


    // Buffer pour les positions
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer pour les normales
    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Buffer pour les indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Buffer pour les texCoords
    glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);



    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Shaders pour le modèle 3D avec illumination
    const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal; // Ajout des normales
    layout (location = 2) in vec2 aTexCoord;
    uniform mat4 Model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 MVP;
    
    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    void main() {
        gl_Position = MVP * vec4(aPos, 1.0);
        FragPos = vec3(Model * vec4(aPos, 1.0));
        // Transposer l'inverse de la matrice modèle pour les normales
        Normal = mat3(transpose(inverse(Model))) * aNormal;
        TexCoord = aTexCoord;
    }
)";

    const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    out vec4 FragColor;
    
    // Paramètres d'éclairage
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;

    uniform float ambientStrength;
    uniform float specularStrength;
    uniform float shininess;
    uniform sampler2D texture1;
    void main() {
               vec3 texColor = texture(texture1, TexCoord).rgb;
               vec3 ambient = ambientStrength * lightColor * texColor;
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor * texColor;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = ambient + diffuse + specular;
           // FragColor = vec4(result, 1.0);
            FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
    }
)";

    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Paramètres d'éclairage par défaut
    glm::vec3 lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor = glm::vec3(0.7f, 0.7f, 0.7f);
    float ambientStrength = 0.3f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    bool succ=false;
    // Boucle de rendu
    while (!glfwWindowShouldClose(window)) {
        // Effacer les buffers de couleur et de profondeur
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        processInput();
        if (succ == false) 
        {
            if (poseFound == false) {
                cameraPos = glm::vec3(
                    cameraDistance * sin(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX)),
                    cameraDistance * sin(glm::radians(cameraAngleX)),
                    cameraDistance * cos(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX))
                );

                cameraPos += cameraTarget;

            }
            else {
                cameraPos = newview * glm::vec4(cameraPos, 1);
				succ = true;
            }
        }
        view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        
        
        if (modelbool==false)
        {
            // Matrice Model pour l'objet 3D
            Model = glm::mat4(1.0f);
            // Model = translate(Model, glm::vec3(0, 0, cameraDistance));
            Model = rotate(Model, glm::radians(objectRotationX), glm::vec3(1, 0, 0));
            Model = rotate(Model, glm::radians(objectRotationY), glm::vec3(0, 1, 0));
            Model = scale(Model, glm::vec3(.8, .8, .8));
           
        }

        MVP = projection * view * Model;

        // Dessiner d'abord l'objet 3D
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "Model"), 1, GL_FALSE, &Model[0][0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

        // Passer les paramètres d'éclairage au shader
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), ambientStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), specularStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), shininess);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        // Échange des buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Nettoyage
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glDeleteVertexArrays(1, &overlayVAO);
    glDeleteBuffers(1, &overlayVBO);
    glDeleteBuffers(1, &overlayEBO);
    glDeleteBuffers(1, &normalVBO);
    glDeleteBuffers(1, &texCoordVBO);
    glDeleteProgram(overlayShaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
