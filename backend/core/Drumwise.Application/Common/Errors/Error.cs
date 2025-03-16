using System.Resources;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Application.Common.Errors;

public class Error
{
    private static readonly ResourceManager ResourceManager = new ResourceManager(typeof(ErrorMessages));
    
    public required string Code { get; set; }
    public required string Description { get; set; }

    private Error()
    {
    }

    private Error(string errorCode)
    {
    }
    
    public static IEnumerable<Error> None => Enumerable.Empty<Error>();

    public static IEnumerable<Error> ApplicationError(IEnumerable<string> errorCodes, params object?[] additionalDescriptionElements) => 
        errorCodes
            .Select(errorCode => new Error { Code = errorCode, Description = string.Format(GetErrorMessage(errorCode), additionalDescriptionElements) })
            .ToList();

    public static IEnumerable<Error> IdentityError(IEnumerable<IdentityError> identityErrors) =>
        identityErrors
            .Select(identityError => new Error { Code = identityError.Code, Description = identityError.Description })
            .ToList();
    
    public static string GetErrorMessage(string errorCode) =>
        ResourceManager.GetString(errorCode) ?? "Unknown error";
}