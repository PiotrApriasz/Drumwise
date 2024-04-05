namespace Drumwise.Application.Common.Interfaces;

public interface IFileService
{
    Task<string> GetTemplateAsync(string templateType, string templateName);
}