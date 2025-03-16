using Drumwise.Application.Common.Models.Settings;

namespace Drumwise.Application.Common.Interfaces;

public interface IFileStorageService
{
    Task<(bool, string)> SaveFileAsync(Stream file, string fileName, string fileType, CancellationToken cancellationToken);
    Task<bool> GetFileAsync(string fileId, string savePath, CancellationToken cancellationToken);
}