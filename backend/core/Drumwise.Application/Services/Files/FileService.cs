using System.Reflection;
using Drumwise.Application.Common.Interfaces;

namespace Drumwise.Application.Services.Files;

public class FileService : IFileService
{
    public async Task<string> GetTemplateAsync(string templateType, string templateName)
    {
        var assembly = Assembly.GetAssembly(typeof(FileService));
        var resourcePath = $"Drumwise.Application.Templates.{templateType}.{templateName}";

        await using var stream = assembly!.GetManifestResourceStream(resourcePath);
        using var reader = new StreamReader(stream!);
        return await reader.ReadToEndAsync();
    }
}